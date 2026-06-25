# AutoDL 单卡实验操作单

## 1. 本地 Git 提交与推送

在本机项目目录执行：

```bash
cd C:/Users/26632/Documents/jimin.xiao3/aaai2027-rpc-clip
git add .
git commit -m "Add RPC-CLIP AAAI 2027 scaffold"
git branch -M main
git remote add origin <你的Git仓库地址>
git push -u origin main
```

如果你已经添加过 remote：

```bash
git remote -v
git push
```

## 2. AutoDL 租卡建议

- GPU：优先 4090 / 3090 / A5000 / A6000，单卡即可。
- 镜像：选择 PyTorch 2.x + CUDA 11.8/12.x 的官方环境。
- 系统盘：30GB 以上。
- 数据盘：80GB 以上，VOC 很小，但后续如果加 COCO 会更稳。
- 训练速度优先：先跑 VOC fast preset，也就是默认 8 epochs。

## 3. AutoDL 上拉代码和装环境

```bash
cd /root/autodl-tmp
git clone <你的Git仓库地址> aaai2027-rpc-clip
cd aaai2027-rpc-clip
bash scripts/autodl_setup.sh
```

如果 `open_clip_torch` 下载失败，直接重跑：

```bash
pip install open_clip_torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## 4. 下载 VOC 2012

```bash
bash scripts/download_voc.sh /root/autodl-tmp/datasets
```

下载后默认路径应为：

```bash
/root/autodl-tmp/datasets/VOCdevkit/VOC2012
```

如果你把数据放到别处，改 `configs/voc_rpc_clip.yaml` 里的 `data.root`。

## 5. 主实验

```bash
bash scripts/run_voc_single_gpu.sh
```

输出目录：

```bash
runs/rpc_clip_voc
```

重点看这些文件：

- `runs/rpc_clip_voc/metrics.csv`
- `runs/rpc_clip_voc/eval.jsonl`
- `runs/rpc_clip_voc/best.pt`
- `runs/rpc_clip_voc/visuals/*.png`
- `runs/rpc_clip_voc/config.yaml`

如果显存不够：

```bash
python train.py --config configs/voc_rpc_clip.yaml --output runs/rpc_clip_voc_bs8 --set data.batch_size=8
```

如果第一轮速度和效果都还可以，再跑更长版本：

```bash
python train.py --config configs/voc_rpc_clip.yaml --output runs/rpc_clip_voc_20ep --set train.epochs=20
python evaluate.py --config runs/rpc_clip_voc_20ep/config.yaml --checkpoint runs/rpc_clip_voc_20ep/best.pt --output runs/rpc_clip_voc_20ep
python visualize.py --config runs/rpc_clip_voc_20ep/config.yaml --checkpoint runs/rpc_clip_voc_20ep/best.pt --output runs/rpc_clip_voc_20ep/visuals --limit 32
```

## 6. 消融实验

```bash
bash scripts/run_ablations.sh
```

输出目录：

```bash
runs/ablations/text_only
runs/ablations/text_proto
runs/ablations/text_proto_reliable
runs/ablations/full_rpc_clip
```

把每个目录里的 best val mIoU 填到：

```bash
experiments/ablation_template.csv
```

## 7. 打包结果给我继续调

主实验结束后：

```bash
bash scripts/package_results.sh runs/rpc_clip_voc
```

消融实验结束后也可以单独打包：

```bash
zip -r rpc_clip_ablations.zip runs/ablations experiments configs paper README.md docs
```

把 zip 发回来后，我会看：

- 训练 loss 是否正常下降。
- 伪标签/预测图是否过度背景化或过度前景化。
- 哪些类别 mIoU 最差。
- 消融是否支持论文里的方法贡献。
- 是否需要改阈值、prompt、prototype momentum、输入分辨率或 loss 权重。

## 8. 填论文表格

主表填 `paper/main.tex` 的 comparison table。

消融表填 `paper/main.tex` 的 ablation table。

Markdown 预览同步改：

```bash
paper/main.md
```

实验记录同步写：

```bash
experiments/experiment_log.md
```

