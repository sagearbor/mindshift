import Foundation

/// A frame the server sends down the live-session socket.
///
/// `server/watch/routers/ws.py:261`, `:271`, `:281`, `:391`, `:434-485`, `:377`/`:489`.
/// Every frame the server sends is modelled (the same set `EpisodeWsClient.kt`
/// dispatches); anything unknown decodes to `.other` rather than failing, the
/// same posture as Kotlin's `ignoreUnknownKeys`.
enum ServerFrame {
    /// `{"type":"nudge", channel, level, t, vectors}` — feeds the escalation
    /// ladder and the reminder.
    case nudge(channel: String, level: Int, t: Double, vectors: [String])
    /// `{"type":"positive", code, t}` — D / E / R. Carries **no channel and no
    /// level** on purpose (`WireModels.kt:40-55`): a positive must never move a
    /// level and must never be repeated by the reminder.
    case positive(code: String, t: Double)
    /// `{"type":"vector_event", vector, level, t, value, detail}`.
    case vectorEvent(vector: String, level: Int, t: Double)
    /// The server's ack of our `{"type":"companion"}` hello (`ws.py:391`).
    case companionAck
    /// `{"type":"live_session_saved", live_session_id, status}` — the answer to
    /// `{"type":"end"}`. `status` is "companion" (nothing persisted, by design),
    /// "companion_hr" (only heart rate kept) or "captured" (a real mic episode);
    /// `nil` only if the server omitted it. Was decoded as `.other` and dropped
    /// until 2026-09-25 — the same bug the Wear client had (`EpisodeWsClient.kt`).
    case liveSessionSaved(id: String, status: String?)
    /// `{"type":"error", detail}` — `malformed_json` or `unknown_type` today. The
    /// socket stays open afterwards (`ws.py` `continue`s), so this is a message,
    /// not a drop, and it is the ONLY way the server says the watch's frames are
    /// wrong (an older server answers the companion hello with one).
    case error(detail: String)
    case other(type: String)

    static func decode(_ text: String) -> ServerFrame? {
        guard let data = text.data(using: .utf8),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              let type = obj["type"] as? String
        else { return nil }
        switch type {
        case "nudge":
            return .nudge(
                channel: obj["channel"] as? String ?? "A",
                level: (obj["level"] as? NSNumber)?.intValue ?? 0,
                t: (obj["t"] as? NSNumber)?.doubleValue ?? 0,
                vectors: obj["vectors"] as? [String] ?? []
            )
        case "positive":
            guard let code = obj["code"] as? String else { return .other(type: type) }
            return .positive(code: code, t: (obj["t"] as? NSNumber)?.doubleValue ?? 0)
        case "vector_event":
            guard let vector = obj["vector"] as? String else { return .other(type: type) }
            return .vectorEvent(
                vector: vector,
                level: (obj["level"] as? NSNumber)?.intValue ?? 0,
                t: (obj["t"] as? NSNumber)?.doubleValue ?? 0
            )
        case "companion_ack":
            return .companionAck
        case "live_session_saved":
            guard let id = obj["live_session_id"] as? String else { return .other(type: type) }
            return .liveSessionSaved(id: id, status: obj["status"] as? String)
        case "error":
            return .error(detail: obj["detail"] as? String ?? "unknown")
        default:
            return .other(type: type)
        }
    }
}

/// COMPANION mode's socket: the phone listens, the wrist only receives relayed
/// nudges and renders them (`ModePolicy.kt:9`, `:21-24`, `:38`).
///
/// This is the highest-fidelity mode watchOS can host (spec §10.3): no mic, no
/// PCM, no trigger evaluation, no duty cycle. The watch is a socket and a buzzer
/// — which is the one thing a watchOS app can nearly do.
///
/// **The supersession hazard is real here.** `EpisodeWsClient` holds no reconnect
/// state and is created one-per-episode (`EpisodeWsClient.kt:76-79`), with the
/// controller minting a listener generation token so a superseded client's
/// callbacks self-neuter (`SentinelController.kt:984-997`).
/// `URLSessionWebSocketTask` has exactly the same hazard — a receive callback
/// from a task you already replaced will happily deliver a nudge — so the same
/// pattern is reproduced: every socket carries a `generation`, and every
/// callback is dropped unless its generation is still current.
actor CompanionSocket {
    /// `SentinelController.kt:1163` — matches the OkHttp `pingInterval` on Wear.
    static let heartbeatIntervalMs: Int64 = 20_000

    enum Event {
        case opened
        case frame(ServerFrame)
        case closed(reason: String)
    }

    private var task: URLSessionWebSocketTask?
    private var generation = 0
    private let session: URLSession
    private var handler: (@Sendable (Event) -> Void)?
    private var heartbeatTask: Task<Void, Never>?

    init(session: URLSession = .shared) {
        self.session = session
    }

    var isOpen: Bool { task?.state == .running }

    /// Opens a socket for `sessionId`. Any previous socket is cancelled and its
    /// generation retired, so nothing it delivers later is acted on.
    func open(
        wsBaseUrl: URL,
        sessionId: String,
        token: String,
        onEvent: @escaping @Sendable (Event) -> Void
    ) {
        closeInternal(reason: "superseded")
        generation += 1
        let myGeneration = generation
        handler = onEvent

        var components = URLComponents(
            url: wsBaseUrl.appendingPathComponent("ws/live-session/\(sessionId)"),
            resolvingAgainstBaseURL: false
        )!
        // `EpisodeWsClient.kt:83` — the token rides as a query param because a
        // WebSocket handshake cannot carry arbitrary headers portably.
        components.queryItems = [URLQueryItem(name: "token", value: token)]
        guard let url = components.url else {
            onEvent(.closed(reason: "bad url"))
            return
        }

        let socket = session.webSocketTask(with: url)
        task = socket
        socket.resume()

        // Tier B hello: tells the server to persist nothing — no live-session
        // doc on `end`, no `not_analyzed` fallback save on an abrupt disconnect,
        // because an all-day wrist socket would otherwise mint an empty junk doc
        // per drop (`ws.py:249-256`).
        send(text: #"{"type":"companion"}"#)
        onEvent(.opened)
        receive(generation: myGeneration)
        startHeartbeat(generation: myGeneration)
    }

    func close(reason: String = "disarm") {
        closeInternal(reason: reason)
    }

    private func closeInternal(reason: String) {
        heartbeatTask?.cancel()
        heartbeatTask = nil
        task?.cancel(with: .goingAway, reason: Data(reason.utf8))
        task = nil
        // Retiring the generation is what neuters any in-flight callback from
        // the socket we just dropped.
        generation += 1
    }

    private func send(text: String) {
        task?.send(.string(text)) { _ in }
    }

    /// `{"type":"end"}` — episode completion. A companion socket that collected
    /// nothing persists nothing (`ws.py:426-441`).
    func end() {
        send(text: #"{"type":"end"}"#)
    }

    private func receive(generation myGeneration: Int) {
        guard let socket = task else { return }
        socket.receive { [weak self] result in
            guard let self else { return }
            Task { await self.handleReceive(result, generation: myGeneration) }
        }
    }

    private func handleReceive(
        _ result: Result<URLSessionWebSocketTask.Message, Error>,
        generation myGeneration: Int
    ) {
        // The supersession gate. Without it, a receive that was already in
        // flight when the socket was replaced delivers a nudge from a dead
        // connection.
        guard myGeneration == generation, let handler else { return }
        switch result {
        case .success(let message):
            switch message {
            case .string(let text):
                if let frame = ServerFrame.decode(text) { handler(.frame(frame)) }
            case .data:
                break // The wrist never receives binary in companion mode.
            @unknown default:
                break
            }
            receive(generation: myGeneration)
        case .failure(let error):
            handler(.closed(reason: error.localizedDescription))
        }
    }

    /// `{"type":"heartbeat"}` every 20 s (`SentinelController.kt:581-604`,
    /// `HEARTBEAT_INTERVAL_MS` at `:1163`). The server answers nothing; the
    /// point is keeping the relay registration alive through NAT and Cloud Run.
    private func startHeartbeat(generation myGeneration: Int) {
        heartbeatTask = Task { [weak self] in
            let nanos = UInt64(Self.heartbeatIntervalMs) * 1_000_000
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: nanos)
                guard let self, await self.generation == myGeneration else { return }
                await self.send(text: #"{"type":"heartbeat"}"#)
            }
        }
    }
}
