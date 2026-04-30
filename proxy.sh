#!/bin/bash

# --- 配置区域 ---
# 在这里定义你的代理地址
PROXY_ADDR="192.168.164.197:7890"
# ---------------

function proxy_on() {
    export http_proxy="http://$PROXY_ADDR"
    export https_proxy="http://$PROXY_ADDR"
    export all_proxy="socks5://$PROXY_ADDR"
    echo -e "\033[32m[✓] 代理已开启: $PROXY_ADDR\033[0m"
}

function proxy_off() {
    unset http_proxy
    unset https_proxy
    unset all_proxy
    echo -e "\033[31m[✗] 代理已关闭\033[0m"
}

function proxy_status() {
    if [ -z "$http_proxy" ]; then
        echo "当前状态: 未开启代理"
    else
        echo "当前状态: 代理运行中 ($http_proxy)"
    fi
}

# 简单的交互逻辑
case "$1" in
    on)  proxy_on ;;
    off) proxy_off ;;
    st)  proxy_status ;;
    *)   echo "用法: source proxy.sh [on|off|st]" ;;
esac