import AVFoundation
import ExpoModulesCore

private let routeChangeEvent = "onRouteChange"

/// Reports the current audio OUTPUT route for Live Coach earpiece mode
/// (apps/mobile/src/live/audioRoute.ts classifies it). Read-only: it never
/// changes the session category or routing.
/// Snapshot: platform "ios", outputs/active = currentRoute.outputs port types
/// (AVAudioSession.Port raw values), communication/mode = nil.
public class AudioRouteModule: Module {
  private var observer: NSObjectProtocol?

  public func definition() -> ModuleDefinition {
    Name("MindShiftAudioRoute")

    Events(routeChangeEvent)

    Function("getRoute") { () -> [String: Any?] in
      return AudioRouteModule.snapshot()
    }

    OnStartObserving(routeChangeEvent) {
      self.startWatching()
    }

    OnStopObserving(routeChangeEvent) {
      self.stopWatching()
    }

    OnDestroy {
      self.stopWatching()
    }
  }

  private func startWatching() {
    if observer != nil { return }
    observer = NotificationCenter.default.addObserver(
      forName: AVAudioSession.routeChangeNotification,
      object: nil,
      queue: .main
    ) { [weak self] _ in
      self?.sendEvent(routeChangeEvent, AudioRouteModule.snapshot())
    }
  }

  private func stopWatching() {
    if let observer = observer {
      NotificationCenter.default.removeObserver(observer)
    }
    observer = nil
  }

  static func snapshot() -> [String: Any?] {
    let outputs = AVAudioSession.sharedInstance().currentRoute.outputs.map { $0.portType.rawValue }
    return [
      "platform": "ios",
      "outputs": outputs,
      "active": outputs,
      "communication": nil,
      "mode": nil,
    ]
  }
}
