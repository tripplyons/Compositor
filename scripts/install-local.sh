#!/bin/bash
# Builds Release without a signing certificate and installs it in /Applications.
# An ad-hoc signature has no team, so the hardened runtime refuses to load the embedded Sparkle framework;
# the local copy is re-signed with library validation off. Release builds signed with Developer ID don't need this.
set -euo pipefail
cd "$(dirname "$0")/.."

derived=/tmp/compositor-release
app=/Applications/Compositor.app
entitlements=$(mktemp -t compositor-local).plist

xcodebuild build -project Compositor.xcodeproj -scheme Compositor -configuration Release \
    -derivedDataPath "$derived" CODE_SIGN_IDENTITY=- CODE_SIGN_STYLE=Manual DEVELOPMENT_TEAM= -quiet

rm -rf "$app"
ditto "$derived/Build/Products/Release/Compositor.app" "$app"
codesign -d --entitlements "$entitlements" --xml "$app" 2>/dev/null
/usr/libexec/PlistBuddy -c "Add :com.apple.security.cs.disable-library-validation bool true" "$entitlements"
codesign -f -s - -o runtime --entitlements "$entitlements" "$app"
codesign --verify --deep --strict "$app"
echo "Installed $app"
