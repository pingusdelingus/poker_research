#!/bin/bash

# Build the project
cd build
make -j4 poker_bot || exit 1
cd ..

# On modern macOS, 'leaks' cannot attach to a process unless it has the get-task-allow entitlement
echo "Granting get-task-allow entitlement to binary for macOS leaks compatibility..."
cat << 'EOF' > get_task_allow.plist
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>com.apple.security.get-task-allow</key>
    <true/>
</dict>
</plist>
EOF

codesign -s - -f --entitlements get_task_allow.plist build/poker_bot

echo "Starting poker_bot with memory leak tracking (MallocStackLogging=1)..."
export MallocStackLogging=1
echo "2" | ./build/poker_bot > /dev/null 2>&1 &
BOT_PID=$!

echo "Bot PID is $BOT_PID. Waiting for it to run for 10 seconds..."
sleep 10

echo "Running macOS 'leaks' utility..."
leaks $BOT_PID > leaks_output.txt
LEAKS_EXIT_CODE=$?

echo "Terminating poker_bot..."
kill -INT $BOT_PID

rm get_task_allow.plist

if [ $LEAKS_EXIT_CODE -eq 0 ]; then
    echo "SUCCESS: No memory leaks detected!"
else
    echo "FAILURE: Memory leaks detected! See leaks_output.txt for details."
    grep -E "Leak:|leaks Report Version" leaks_output.txt | head -n 10
fi

exit $LEAKS_EXIT_CODE
