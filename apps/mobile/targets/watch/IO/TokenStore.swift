import Foundation
import Security

/// Where the device token lives.
///
/// The Wear app keeps it in plain `SharedPreferences` — file `"gauge_account"`,
/// keys `"account_id"` / `"device_token"`, unencrypted, no Keystore
/// (`auth/AccountPrefs.kt:21-23`). On watchOS the Keychain is present and free,
/// so this is one of the few places the port is a straight *improvement* over
/// the shipped app rather than a degradation (spec §11.3, rated FULL).
///
/// Split deliberately: the 256-bit bearer token goes in the Keychain, the
/// non-secret `account_id` in `UserDefaults`. Putting the account id in the
/// Keychain too would buy nothing and make it harder to read from a widget.
struct TokenStore {
    private static let service = "com.sagearbor.mindshift.watch"
    private static let tokenAccount = "device_token"
    private static let accountIdKey = "mindshift.account_id"

    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) { self.defaults = defaults }

    var accountId: String? {
        get { defaults.string(forKey: Self.accountIdKey) }
        nonmutating set {
            if let newValue { defaults.set(newValue, forKey: Self.accountIdKey) }
            else { defaults.removeObject(forKey: Self.accountIdKey) }
        }
    }

    var deviceToken: String? {
        get { Self.readKeychain() }
        nonmutating set {
            if let newValue { Self.writeKeychain(newValue) } else { Self.deleteKeychain() }
        }
    }

    var isSignedIn: Bool { accountId != nil && deviceToken != nil }

    func save(accountId: String, deviceToken: String) {
        self.accountId = accountId
        self.deviceToken = deviceToken
    }

    /// A 401 clears the token and routes back to sign-in — the watch must KNOW
    /// it is signed out rather than retrying forever (`control/CouplesLoad.kt:19-24`).
    ///
    /// A 503 must NOT come here. The server deliberately answers 503 rather than
    /// 401 for a transient verifier outage precisely so the watch does not sign
    /// itself out on a Firestore blip (`server/watch/auth.py:156-162`, `:263-277`).
    /// Getting this backwards logs the user out on every server hiccup, which is
    /// why `WatchApi` branches on the status code and only 401 reaches here.
    func clearForUnauthorized() {
        accountId = nil
        deviceToken = nil
    }

    // MARK: - Keychain

    private static func baseQuery() -> [String: Any] {
        [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: tokenAccount,
        ]
    }

    private static func readKeychain() -> String? {
        var query = baseQuery()
        query[kSecReturnData as String] = true
        query[kSecMatchLimit as String] = kSecMatchLimitOne
        var item: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &item) == errSecSuccess,
              let data = item as? Data,
              let token = String(data: data, encoding: .utf8),
              !token.isEmpty
        else { return nil }
        return token
    }

    private static func writeKeychain(_ token: String) {
        deleteKeychain()
        var query = baseQuery()
        query[kSecValueData as String] = Data(token.utf8)
        // The watch is unlocked whenever it is on the wrist, and the socket has
        // to be able to reconnect with the screen off, so ...AfterFirstUnlock
        // rather than ...WhenUnlocked.
        query[kSecAttrAccessible as String] = kSecAttrAccessibleAfterFirstUnlock
        SecItemAdd(query as CFDictionary, nil)
    }

    private static func deleteKeychain() {
        SecItemDelete(baseQuery() as CFDictionary)
    }
}
