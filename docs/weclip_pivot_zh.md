# WeCLIP Pivot 操作单

## 为什么要转向 WeCLIP

当前 RPC-CLIP 自研原型已经完成了四轮有效诊断：

- v1/v2：背景坍塌，background IoU 为 0。
- v3：显式背景后，mIoU 约 2.95%，但伪标签监督过密。
- v4：保守 quantile seed 后，mIoU 约 3.9%，伪标签比例正常。
- v5：joint inference 提升 background IoU，但前景被压没。
- v6：QuickGELU 匹配后仍约 3.9%，说明瓶颈不是 CLIP 激活函数。

结论：这个从零搭的小 decoder 路线不适合继续作为 AAAI 主线。下一步应以
WeCLIP/Frozen CLIP 这类团队已有强基线为基础做轻量方法改动，再写论文。

## AutoDL 克隆 WeCLIP

```bash
cd /root/autodl-tmp/aaai2027-rpc-clip
bash scripts/clone_weclip_autodl.sh
```

如果 GitHub 仍失败，用代理或上传 zip：

```bash
cd /root/autodl-tmp
git clone --depth 1 https://gh-proxy.com/https://github.com/zbf1991/WeCLIP.git WeCLIP
```

## 推荐下一步

1. 先按 WeCLIP 官方 README 跑通 VOC 复现。
2. 记录 baseline 的 val/test mIoU、训练时间和显存。
3. 在 WeCLIP 的伪标签/refinement 阶段加入我们已经诊断过的可靠性思路：
   - quantile background seeds；
   - entropy reliability；
   - prototype momentum calibration；
   - failure-case visualization。
4. 论文从“新模型从零训练”改成“对 Frozen CLIP WSSS 的可靠伪标签校准模块”。

## 给我回传什么

跑通 WeCLIP baseline 后，把以下内容打包：

```bash
zip -r weclip_baseline_results.zip \
  /root/autodl-tmp/WeCLIP/WeCLIP/exp \
  /root/autodl-tmp/WeCLIP/WeCLIP/logs \
  /root/autodl-tmp/WeCLIP/WeCLIP/*.sh
```

如果路径不存在，先执行：

```bash
find /root/autodl-tmp/WeCLIP -maxdepth 4 -type f | head -100
```

把输出和结果 zip 发回来，我再把方法改动插到正确位置。

