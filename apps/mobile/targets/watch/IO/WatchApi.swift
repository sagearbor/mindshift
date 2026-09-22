import Foundation

// MARK: - Wire models

/// `POST /me/pair/start` response — `server/watch/routers/pairing.py:243-246`,
/// mirrored in Kotlin at `shared/.../WireModels.kt:290-294`.
struct PairingStart: Decodable {
    let code: String
    let pairingId: String
    let expiresAt: String

    enum CodingKeys: String, CodingKey {
        case code
        case pairingId = "pairing_id"
        case expiresAt = "expires_at"
    }
}

/// `GET /me/pair/status` response — `pairing.py:249-252`, `WireModels.kt:299-303`.
///
/// The endpoint **always answers HTTP 200**, and reports unknown and expired
/// pairings both as `status: "expired"` on purpose (`pairing.py:291-299`), so a
/// decoded value here is always a real lifecycle state.
struct PairingStatus: Decodable {
    let status: String
    let accountId: String?
    let deviceToken: String?

    enum CodingKeys: String, CodingKey {
        case status
        case accountId = "account_id"
        case deviceToken = "device_token"
    }
}

/// The slice of `GET /me/standing` this screen shows. `ignoreUnknownKeys` is the
/// Kotlin default (`WireModels.kt:21`) and `Decodable` ignores unknown keys the
/// same way, so the server may keep adding fields.
struct MemberStanding: Decodable {
    let accountId: String?
    let displayName: String?
    let sessions: Int?
    let calmScore: Double?

    enum CodingKeys: String, CodingKey {
        case accountId = "account_id"
        case displayName = "display_name"
        case sessions
        case calmScore = "calm_score"
    }
}

// MARK: - Client

enum WatchApiError: Error, Equatable {
    case notSignedIn
    case transport(String)
    /// 401 — the token is dead. The caller must clear it and route to sign-in.
    case unauthorized
    /// 503 — the verifier is having a moment. The caller must **keep** the
    /// token: `server/watch/auth.py:156-162` returns 503 rather than 401 for a
    /// transient outage specifically so the watch does not sign itself out.
    case serverUnavailable
    case http(Int)
    case decode(String)
}

/// The watch's HTTP client: two unauthenticated pairing calls, one bearer-token
/// call.
///
/// Mirrors `DevicePairingClient.kt`'s shape, including the parts that were
/// tuned against production rather than chosen:
/// * **no `Authorization` header on either pairing call, by design**
///   (`DevicePairingClient.kt:26-28`) — `pairing_id` *is* the capability;
/// * `connectTimeout` 5 s (`:53`), overall `callTimeout` 30 s (`:126`, raised
///   from 10 s for Cloud Run cold starts);
/// * **exactly one silent retry, transport errors only** (`:63-68`). A 4xx/5xx
///   is an answer and is never retried.
actor WatchApi {
    /// `apps/watch/wearApp/build.gradle.kts:28-32`.
    static let defaultBaseUrl = URL(string: "https://mindshift-api-664594784582.us-central1.run.app")!

    private let baseUrl: URL
    private let session: URLSession
    private let decoder = JSONDecoder()

    init(baseUrl: URL = WatchApi.defaultBaseUrl) {
        self.baseUrl = baseUrl
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30 // callTimeout 30 s
        config.timeoutIntervalForResource = 30
        config.waitsForConnectivity = false
        self.session = URLSession(configuration: config)
    }

    var wsBaseUrl: URL {
        var components = URLComponents(url: baseUrl, resolvingAgainstBaseURL: false)!
        components.scheme = baseUrl.scheme == "http" ? "ws" : "wss"
        return components.url!
    }

    // MARK: Pairing

    func startPairing() async throws -> PairingStart {
        var request = URLRequest(url: baseUrl.appendingPathComponent("me/pair/start"))
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = Data("{}".utf8)
        return try await perform(request, as: PairingStart.self)
    }

    func pollPairing(pairingId: String) async throws -> PairingStatus {
        var components = URLComponents(
            url: baseUrl.appendingPathComponent("me/pair/status"),
            resolvingAgainstBaseURL: false
        )!
        components.queryItems = [URLQueryItem(name: "pairing_id", value: pairingId)]
        return try await perform(URLRequest(url: components.url!), as: PairingStatus.self)
    }

    // MARK: Authenticated

    func standing(token: String) async throws -> MemberStanding {
        var request = URLRequest(url: baseUrl.appendingPathComponent("me/standing"))
        // `net/WatchApiClient.kt:124-128`.
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        return try await perform(request, as: MemberStanding.self)
    }

    /// Fire-and-forget, exactly as `SignInScreen.kt:88-95` does it: a failure
    /// here must never block a successful sign-in.
    func claimLegacy(token: String) async {
        var request = URLRequest(url: baseUrl.appendingPathComponent("me/claim-legacy"))
        request.httpMethod = "POST"
        request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        _ = try? await session.data(for: request)
    }

    // MARK: Transport

    private func perform<T: Decodable>(_ request: URLRequest, as type: T.Type) async throws -> T {
        let data: Data
        let response: URLResponse
        do {
            (data, response) = try await session.data(for: request)
        } catch {
            // The one silent retry (`DevicePairingClient.kt:63-68`) — transport
            // only. Cloud Run cold starts are the reason it exists.
            do {
                (data, response) = try await session.data(for: request)
            } catch let retryError {
                throw WatchApiError.transport(retryError.localizedDescription)
            }
        }
        guard let http = response as? HTTPURLResponse else {
            throw WatchApiError.transport("no HTTP response")
        }
        switch http.statusCode {
        case 200..<300:
            do {
                return try decoder.decode(type, from: data)
            } catch {
                throw WatchApiError.decode(String(describing: error))
            }
        case 401:
            throw WatchApiError.unauthorized
        case 503:
            throw WatchApiError.serverUnavailable
        default:
            throw WatchApiError.http(http.statusCode)
        }
    }
}
