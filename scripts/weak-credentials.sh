# Shared weak-credential denylist for the identity-provider bootstrap paths.
#
# Single source of truth, sourced by both:
#   - pingfederate/setup/init-pingfederate.sh  (fail-closed gate, runs post-deploy)
#   - build_and_run.sh                         (preflight, runs pre-deploy)
#
# Two consumers previously kept hand-synced copies that drifted, which let
# PF_ADMIN_PASS=changeme clear preflight and then abort the bootstrap minutes
# later. Add values here only.
#
# Entries MUST be lowercase: both consumers compare against a lowercased value.
# Shorter than the 12-character floor is redundant but harmless -- a short weak
# value gets the precise "known-weak" error instead of a misleading length error.

WEAK_CREDENTIALS=(
    "changeme" "changeit" "admin123" "2federatem0re" "password" "passw0rd"
    "admin" "administrator" "testuser" "test" "secret" "default"
    "123456" "12345678" "letmein" "welcome" "pingfederate"
    "change-password-to-some-secret-password"
)

# Minimum length for any operator-supplied bootstrap credential.
WEAK_CREDENTIAL_MIN_LEN=12

# Return 0 when $1 is a known-weak value. Trims surrounding whitespace and folds
# case under LC_ALL=C so the ASCII mapping is used regardless of locale (a
# Turkish locale maps I/i differently under tr's [:upper:]/[:lower:] classes).
is_weak_credential() {
    local value="$1"
    local stripped="$value"
    stripped="${stripped#"${stripped%%[![:space:]]*}"}"
    stripped="${stripped%"${stripped##*[![:space:]]}"}"

    local lowered
    lowered="$(printf '%s' "$stripped" | LC_ALL=C tr 'A-Z' 'a-z')"

    local weak
    for weak in "${WEAK_CREDENTIALS[@]}"; do
        if [ "$lowered" = "$weak" ]; then
            return 0
        fi
    done
    return 1
}
