#!/usr/bin/env bash
# gc_9b_watchdog.sh — 9B 实验守护 (独立进程组, 与编排互拉)
# 背景: 11-07 两次事故 (12:23/17:08) 编排进程组被外部按组信号灭杀 (EngineCore 因独立组幸存),
#   本守护 setsid 独立会话运行: 编排死且实验未完 → 每 60s 重拉; 三组齐 → 退出
set -u
G=/root/code/GCRAFT
PIDFILE=/tmp/gcraft_9b.pid

results_done() {
    [ -f $G/results_v2/new_qa/9b/result.txt ] && \
    [ -f /root/code/Table-Critic/results_v2/orig_qa/9b/result.txt ] && \
    [ -f /root/code/Table-Critic/results_v2/orig_fv/9b/result.txt ]
}

echo "[$(date -Is)] 守护启动 (pid $$, setsid 会话)"
while true; do
    if results_done; then
        echo "[$(date -Is)] 三组齐, 守护退出"
        rm -f $PIDFILE
        exit 0
    fi
    alive=false
    if [ -f $PIDFILE ]; then
        kill -0 "$(cat $PIDFILE)" 2>/dev/null && alive=true
    fi
    if ! $alive; then
        echo "[$(date -Is)] 编排不在/已死, 重拉..."
        rm -f $PIDFILE
        cd $G && setsid nohup bash scripts/run_9b_only.sh >> logs/run_9b_only_v2.log 2>&1 < /dev/null &
        echo $! > $PIDFILE
    fi
    sleep 60
done
