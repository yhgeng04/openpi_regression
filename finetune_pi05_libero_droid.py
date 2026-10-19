#!/usr/bin/env python3
r"""将 README 中的 pi05_libero 转成 PyTorch，再在 DROID RLDS 子集上继续微调。

环境（在 Linux / NVIDIA CUDA 服务器的 openpi 根目录运行，Python 3.11）：
    uv sync --python 3.11 --group rlds --link-mode copy
    # 安装 Google Cloud CLI，确保 gsutil 可用；仓库下载器会调用它。

完整流程（数据目录结构必须是 /data/rlds/droid/1.0.1/）：
    uv run --group rlds python finetune_pi05_libero_droid.py \
        --rlds-data-dir /data/rlds --patch-transformers

如果还没有 DROID 数据，额外加 --download-droid（默认最多下载 500 GB）：
    uv run --group rlds python finetune_pi05_libero_droid.py \
        --rlds-data-dir /data/rlds_500gb --download-droid --droid-max-gb 500 \
        --patch-transformers

500 GB 按十进制字节计算，只限制数据文件和元数据，不包含模型、缓存、训练 checkpoint。
脚本按实际文件大小、固定随机种子挑选完整分片，总量略低于预算，不先下载完整数据。
分片清单保存为 droid/1.0.1/openpi_subset.json；统计和训练严格使用这个清单。
下载采用每批最多 32 个显式文件路径，每批检查大小并最多尝试 3 次。
重复执行下载会跳过大小正确的文件。更改预算/随机种子应使用新的数据目录；
已有完整数据可使用 --droid-max-gb 0（下载完整版本约需 1.8 TB）。
子集不保证包含全部任务；同样训练步数会增加重复采样，不保证与完整数据相同的效果。

分阶段运行：
    ... finetune_pi05_libero_droid.py --stage prepare --patch-transformers
    ... finetune_pi05_libero_droid.py --stage stats --rlds-data-dir /data/rlds
    ... finetune_pi05_libero_droid.py --stage train --rlds-data-dir /data/rlds
    # 中断后：同一组参数加 --resume；--num-train-steps 是目标总步数。
    # 仅查看执行计划（无需安装训练依赖，不下载、不创建目录）：
    python3 finetune_pi05_libero_droid.py --dry-run --rlds-data-dir /data/rlds

输出（均位于本脚本所在的 openpi 目录）：
    pi05_libero/model.safetensors             转换后的 LIBERO 初始权重
    pi05_libero/assets/                       原始 LIBERO 归一化统计
    pi05_libero_droid/model.safetensors       最后一步 DROID 微调权重
    pi05_libero_droid/assets/droid/           本次 DROID 归一化统计
    pi05_libero_droid/config.json             模型/动作空间及来源说明
    pi05_libero_droid/checkpoints/run/<step>/ 模型、优化器、训练状态

这是单 GPU 全参数微调，默认 batch=8、100000 步；batch 可调，不保证默认值适合
你的显存。沿用仓库的梯度检查点和 PyTorch 精度策略，无 LoRA、EMA 或混合精度。
完整 DROID 训练代价很大；步数不代表已收敛。--resume 恢复模型/优化器/步数，
与原训练器一样不恢复 RLDS 迭代位置或随机数状态。

权重始终来自 pi05_libero；使用 DROID 的 16 步动作块、离散状态提示以及
7 维关节位置增量 + 1 维绝对夹爪位置（网络仍为 32 维，剩余补零）。这与
pi05_libero 的动作语义不同，因此需要重新训练和评估，而不是直接推理。
不使用 pi05_base/pi05_droid 权重或它们的归一化统计。推理时可调用本文件的
make_config(parse_args(["--rlds-data-dir", "/data/rlds"]))，再将它与
pi05_libero_droid 路径传给 openpi.policies.policy_config.create_trained_policy；
不能直接使用原版 pi05_droid 推理配置（它的动作空间不同）。
"""

import argparse
import dataclasses
import hashlib
import importlib.util
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "pi05_libero"
OUTPUT = ROOT / "pi05_libero_droid"
SOURCE = "gs://openpi-assets/checkpoints/pi05_libero"
DROID_SOURCE = "gs://gresearch/robotics/droid/1.0.1"
SUBSET_MANIFEST = "openpi_subset.json"
# 官方 1.0.1 目录中的数据集名为 droid_101，实际分片以 droid_101-train 开头。
# 同时兼容未带版本后缀的名称；非捕获组保留 [1]/[2] 的分片编号/总数语义。
SHARD_PATTERN = re.compile(r"droid(?:_\d+)?-train\.tfrecord-(\d+)-of-(\d+)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "prepare", "stats", "train"), default="all")
    parser.add_argument("--rlds-data-dir", type=Path, help="包含 droid/1.0.1 的目录")
    parser.add_argument("--source-checkpoint", default=SOURCE, help="官方 URI 或本地 LIBERO JAX/PyTorch checkpoint")
    parser.add_argument("--download-droid", action="store_true", help="使用 gsutil 下载选定的 DROID 1.0.1 分片")
    parser.add_argument("--droid-max-gb", type=float, default=500, help="数据预算（十进制 GB），默认 500；0 表示完整数据")
    parser.add_argument("--patch-transformers", action="store_true", help="应用 README 要求的 transformers 补丁")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-train-steps", type=int, default=100_000)
    parser.add_argument("--save-interval", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--precision", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--max-stats-frames", type=int, default=1_000_000, help="统计采样帧数，可增加到 10000000")
    parser.add_argument("--shuffle-buffer-size", type=int, default=100_000)
    parser.add_argument("--read-workers", type=int, default=8, help="RLDS 并行读取/转换数量")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    positive_fields = (
        "batch_size", "num_train_steps", "save_interval", "max_stats_frames", "shuffle_buffer_size", "read_workers"
    )
    for field in positive_fields:
        if getattr(args, field) <= 0:
            parser.error(f"--{field.replace('_', '-')} 必须大于 0")
    if args.max_stats_frames < 2 or args.warmup_steps < 0 or not 0 < args.learning_rate < math.inf:
        parser.error("统计至少需要 2 帧，warmup >= 0，learning-rate 必须为有限正数")
    if not 0 <= args.droid_max_gb < math.inf:
        parser.error("--droid-max-gb 必须为有限非负数；0 表示完整数据")
    if args.stage != "prepare" and args.rlds_data_dir is None:
        parser.error("此阶段需要 --rlds-data-dir（包含 droid/1.0.1 的目录）")
    if args.rlds_data_dir is not None:
        args.rlds_data_dir = args.rlds_data_dir.expanduser().resolve()
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("本脚本的 RLDS 适配器仅支持单 GPU；请使用 python 启动，不要使用 torchrun")
    return args


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def patch_transformers():
    """逐文件替换以断开 uv 的硬链接，避免改写 uv 缓存中的文件。"""
    import transformers

    if transformers.__version__ != "4.53.2":
        raise RuntimeError("请先安装 pyproject.toml 指定的 transformers==4.53.2")
    target = Path(transformers.__file__).parent
    source = ROOT / "src/openpi/models_pytorch/transformers_replace"
    for file in source.rglob("*.py"):
        destination = target / file.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as temporary:
            temporary.write(file.read_bytes())
            temporary_path = Path(temporary.name)
        temporary_path.chmod(0o644)
        temporary_path.replace(destination)
    print(f"已应用 transformers 补丁：{target}", flush=True)


def check_transformers():
    try:
        from transformers.models.siglip import check

        valid = check.check_whether_transformers_replace_is_installed_correctly()
    except (ImportError, AttributeError):
        valid = False
    if not valid:
        raise RuntimeError("缺少仓库的 transformers 补丁；请添加 --patch-transformers 后重试")


def prepare_checkpoint(args):
    if (BASE / "model.safetensors").is_file() and (BASE / "source.json").is_file():
        previous = json.loads((BASE / "source.json").read_text())
        if previous["source"] != args.source_checkpoint:
            raise ValueError("pi05_libero 已有其他来源的权重，请使用相同 --source-checkpoint")
        print(f"复用已准备的权重：{BASE}", flush=True)
        return
    if BASE.exists() and any(BASE.iterdir()):
        raise FileExistsError(f"{BASE} 已有未由本脚本确认的文件，请移走该目录后再试")
    check_transformers()
    from openpi.shared import download

    source = download.maybe_download(args.source_checkpoint)
    # 转换器通过路径中的 'pi05' 判断 AdaRMSNorm；即使用户的本地目录被改名也兼容。
    with tempfile.TemporaryDirectory(prefix="pi05_convert_", dir=ROOT) as temporary:
        temporary = Path(temporary)
        staged = temporary / "converted"
        if (source / "model.safetensors").is_file():
            staged.mkdir()
            shutil.copy2(source / "model.safetensors", staged / "model.safetensors")
            if (source / "config.json").is_file():
                shutil.copy2(source / "config.json", staged / "config.json")
        else:
            if not (source / "params").is_dir():
                raise FileNotFoundError(f"{source} 中缺少 params/ 或 model.safetensors")
            alias = temporary / "pi05_libero_jax"
            alias.symlink_to(source, target_is_directory=True)
            subprocess.run(
                [sys.executable, str(ROOT / "examples/convert_jax_model_to_pytorch.py"),
                 "--checkpoint-dir", str(alias), "--config-name", "pi05_libero",
                 "--output-path", str(staged), "--precision", args.precision],
                cwd=ROOT, check=True,
            )
        # 仓库转换器查找 parent/assets；发布的 checkpoint 实际通常是 checkpoint/assets。
        assets = source / "assets"
        if not assets.is_dir():
            assets = source.parent / "assets"
        if not assets.is_dir():
            raise FileNotFoundError(f"{source} 缺少 LIBERO assets，无法生成完整初始 checkpoint")
        shutil.copytree(assets, staged / "assets", dirs_exist_ok=True)
        write_json(staged / "source.json", {"source": args.source_checkpoint, "config_name": "pi05_libero"})
        if BASE.exists():
            BASE.rmdir()  # 只允许删除空目录。
        staged.rename(BASE)
    print(f"PyTorch LIBERO checkpoint 已保存：{BASE}", flush=True)


def shard_split(indices):
    """TFDS 4.9.9 原生 shard 切片；合并相邻分片，不改变官方元数据。"""
    ranges = []
    for index in sorted(indices):
        if ranges and ranges[-1][1] == index:
            ranges[-1][1] = index + 1
        else:
            ranges.append([index, index + 1])
    return "+".join(f"train[{start}shard:{end}shard]" for start, end in ranges)


def select_subset(listing, max_bytes, seed):
    """仅根据 gsutil ls -l 文件清单规划下载；不会读取数据内容。"""
    files = []
    for line in listing.splitlines():
        parts = line.split()
        if len(parts) != 3 or not parts[0].isdigit() or not parts[2].startswith(DROID_SOURCE + "/"):
            continue
        name = parts[2][len(DROID_SOURCE) + 1:]
        if "/" not in name:
            files.append({"name": name, "bytes": int(parts[0])})
    metadata = [file for file in files if file["name"].endswith(".json")]
    if not {"dataset_info.json", "features.json"}.issubset({file["name"] for file in metadata}):
        raise RuntimeError("远端目录缺少 dataset_info.json/features.json，不能创建 RLDS 子集")
    shards = sorted((file for file in files if SHARD_PATTERN.fullmatch(file["name"])), key=lambda f: f["name"])
    if not shards:
        examples = [file["name"] for file in files if "tfrecord" in file["name"]][:5]
        raise RuntimeError(
            f"远端目录没有识别到 DROID train TFRecord 分片；已解析 {len(files)} 个文件，"
            f"TFRecord 文件名示例：{examples}。预期 droid_101-train.tfrecord-00000-of-02048 等格式。"
        )
    random.Random(seed).shuffle(shards)
    # 为本地清单预留 1 MB。每个分片完整下载，不截断 TFRecord。
    used = sum(file["bytes"] for file in metadata) + 1_000_000
    selected = []
    for file in shards:
        if used + file["bytes"] <= max_bytes:
            selected.append(file)
            used += file["bytes"]
    if not selected:
        raise ValueError("数据预算不足以容纳一个完整分片和元数据，请增大 --droid-max-gb")
    selected.sort(key=lambda f: f["name"])
    return {
        "source": DROID_SOURCE, "max_bytes": max_bytes, "seed": seed,
        "selected_bytes": sum(file["bytes"] for file in metadata + selected),
        "split": shard_split([int(SHARD_PATTERN.fullmatch(file["name"])[1]) for file in selected]),
        "files": sorted(metadata + selected, key=lambda f: f["name"]),
    }


def load_subset(args):
    manifest_path = args.rlds_data_dir / "droid/1.0.1" / SUBSET_MANIFEST
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (manifest["source"] != DROID_SOURCE or manifest["max_bytes"] != int(args.droid_max_gb * 1e9)
            or manifest["seed"] != args.seed):
        raise ValueError("子集清单的预算/随机种子与本次不一致，请保持原参数或使用新的数据目录")
    names = [file["name"] for file in manifest["files"]]
    if len(set(names)) != len(names) or any(Path(name).name != name for name in names):
        raise ValueError("子集清单包含重复或非法文件名")
    indices = [int(SHARD_PATTERN.fullmatch(name)[1]) for name in names if SHARD_PATTERN.fullmatch(name)]
    if not indices or manifest["split"] != shard_split(indices):
        raise ValueError("子集清单的 TFDS split 与文件列表不一致")
    if sum(file["bytes"] for file in manifest["files"]) + manifest_path.stat().st_size > manifest["max_bytes"]:
        raise ValueError("子集清单超出数据预算")
    return manifest


def missing_subset_files(files, destination):
    return [file for file in files
            if not (destination / file["name"]).is_file()
            or (destination / file["name"]).stat().st_size != file["bytes"]]


def download_subset_files(files, destination):
    """用显式 argv 分批下载；每批验证文件，即使 gsutil 返回 0 也不跳过检查。"""
    pending = missing_subset_files(files, destination)
    batch_size = 32
    for start in range(0, len(pending), batch_size):
        batch = pending[start:start + batch_size]
        remaining = batch
        for attempt in range(1, 4):
            print(
                f"下载批次 {start // batch_size + 1}/{math.ceil(len(pending) / batch_size)}，"
                f"第 {attempt}/3 次尝试，剩余 {len(remaining)} 个文件",
                flush=True,
            )
            # 不依赖 `cp -I` 从标准输入读取长清单；每个源 URI 都是独立参数。
            result = subprocess.run(
                ["gsutil", "-m", "cp", *[f"{DROID_SOURCE}/{file['name']}" for file in remaining], str(destination)],
                stdin=subprocess.DEVNULL, check=False,
            )
            remaining = missing_subset_files(batch, destination)
            if not remaining:
                break
        else:
            examples = ", ".join(file["name"] for file in remaining[:3])
            raise RuntimeError(
                f"当前批次重试 3 次后仍有 {len(remaining)} 个文件缺失或大小不符（gsutil 返回 {result.returncode}）："
                f"{examples}。已完成的文件会保留；重新运行相同的 --download-droid 命令可继续。"
            )


def ensure_dataset(args):
    destination = args.rlds_data_dir / "droid" / "1.0.1"
    manifest = load_subset(args)
    if args.download_droid:
        if shutil.which("gsutil") is None:
            raise RuntimeError("--download-droid 需要 Google Cloud CLI 的 gsutil")
        destination.mkdir(parents=True, exist_ok=True)
        if args.droid_max_gb == 0:
            subprocess.run(["gsutil", "-m", "rsync", "-r", DROID_SOURCE, str(destination)], check=True)
        else:
            if manifest is None:
                if any(destination.iterdir()):
                    raise FileExistsError("子集下载请使用空的数据目录，避免已有数据超出预算")
                listing = subprocess.run(
                    ["gsutil", "ls", "-l", DROID_SOURCE + "/*"], capture_output=True, text=True, check=True
                ).stdout
                manifest = select_subset(listing, int(args.droid_max_gb * 1e9), args.seed)
                write_json(destination / SUBSET_MANIFEST, manifest)
            selected = {file["name"] for file in manifest["files"]}
            extra = [path.name for path in destination.glob("*tfrecord*")
                     if SHARD_PATTERN.fullmatch(path.name) and path.name not in selected]
            if extra:
                raise ValueError("数据目录存在清单之外的分片，请使用独立的子集目录")
            missing = missing_subset_files(manifest["files"], destination)
            print(f"DROID 子集：{manifest['selected_bytes'] / 1e9:.2f} GB，待下载 {len(missing)} 个文件", flush=True)
            if missing:
                download_subset_files(missing, destination)
    if args.droid_max_gb > 0 and manifest is None:
        raise FileNotFoundError("缺少子集清单；请在新目录添加 --download-droid，已有完整数据请加 --droid-max-gb 0")
    if manifest:
        for file in manifest["files"]:
            path = destination / file["name"]
            if not path.is_file() or path.stat().st_size != file["bytes"]:
                raise FileNotFoundError(f"子集文件缺失或大小不符：{path}；添加 --download-droid 补齐")
    if not (destination / "dataset_info.json").is_file() or not any(destination.glob("*tfrecord*")):
        raise FileNotFoundError(f"未找到 DROID RLDS 数据：{destination}；准备数据或添加 --download-droid")


def make_config(args):
    """构造训练和推理共同使用的配置；不修改仓库的全局配置注册表。"""
    from openpi.training import config
    from openpi.training import weight_loaders

    template = config.get_config("pi05_full_droid_finetune")
    return dataclasses.replace(
        template,
        name="checkpoints", exp_name="run", checkpoint_base_dir=str(OUTPUT),
        assets_base_dir=str(OUTPUT / "training_assets"),
        model=dataclasses.replace(template.model, dtype=args.precision, pytorch_compile_mode=None),
        data=dataclasses.replace(
            template.data, rlds_data_dir=str(args.rlds_data_dir),
            assets=config.AssetsConfig(assets_dir=str(OUTPUT / "assets"), asset_id="droid"),
        ),
        weight_loader=weight_loaders.NoOpWeightLoader(),
        pytorch_weight_path=str(BASE), pytorch_training_precision=args.precision,
        lr_schedule=dataclasses.replace(
            template.lr_schedule, warmup_steps=args.warmup_steps,
            peak_lr=args.learning_rate, decay_lr=args.learning_rate,
        ),
        batch_size=args.batch_size, num_train_steps=args.num_train_steps,
        save_interval=args.save_interval, log_interval=10, seed=args.seed,
        ema_decay=None, num_workers=0, wandb_enabled=False, overwrite=False, resume=args.resume,
    )


class WritableActionsDataset:
    """在原地动作转换之前复制 RLDS 动作数组，供统计和训练共用。"""

    def __init__(self, dataset):
        self.dataset = dataset

    def __iter__(self):
        import numpy as np

        for batch in self.dataset:
            # TensorFlow 的 as_numpy_iterator() 可返回只读数组；DroidInputs 中的
            # np.asarray 不会复制，随后 DeltaActions 的 -= 就会失败。
            # 每批独立复制动作也避免修改源数据；图像等大数组保留原引用。
            yield {**batch, "actions": np.array(batch["actions"], copy=True)}

    def __len__(self):
        return len(self.dataset)


def raw_dataset(args, config, data_config, *, training):
    import dlimp as dl
    import tensorflow as tf

    from openpi.training.droid_rlds_dataset import DroidRldsDataset

    tf.config.set_visible_devices([], "GPU")
    tf.random.set_seed(args.seed)
    manifest = load_subset(args)
    original_from_rlds = dl.DLataset.from_rlds

    def from_subset(builder, *, split="train", **kwargs):
        if split != "train":
            raise ValueError(f"子集适配器不支持 split={split}")
        expected = {file["name"] for file in manifest["files"] if SHARD_PATTERN.fullmatch(file["name"])}
        actual = {Path(item.filename).name for item in builder.info.splits[manifest["split"]].file_instructions}
        if actual != expected:
            raise ValueError("TFDS 实际读取分片与下载清单不同，停止以避免读取缺失文件")
        return original_from_rlds(builder, split=manifest["split"], **kwargs)

    # DROID 原加载器硬编码 split='train'。仅在创建当前数据管线时替换入口，随后恢复。
    # TFDS 4.9.9 支持原生 shard 切片：
    # https://github.com/tensorflow/datasets/blob/v4.9.9/tensorflow_datasets/core/splits.py
    if manifest is not None:
        dl.DLataset.from_rlds = staticmethod(from_subset)
    try:
        dataset = DroidRldsDataset(
            data_dir=str(args.rlds_data_dir), batch_size=args.batch_size,
            datasets=data_config.datasets, action_chunk_size=config.model.action_horizon,
            action_space=data_config.action_space, shuffle=training,
            shuffle_buffer_size=args.shuffle_buffer_size if training else 1,
            num_parallel_reads=args.read_workers, num_parallel_calls=args.read_workers,
        )
        return WritableActionsDataset(dataset)
    finally:
        dl.DLataset.from_rlds = staticmethod(original_from_rlds)


def stats_signature(args):
    signature = {"rlds_data_dir": str(args.rlds_data_dir), "version": "1.0.1", "action_horizon": 16,
                 "actions": "delta_joint_position_7_absolute_gripper_1", "max_frames": args.max_stats_frames,
                 "seed": args.seed, "filter": "droid_sample_ranges_v1_0_1.json"}
    manifest = load_subset(args)
    if manifest is not None:
        signature["subset_sha256"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return signature


def compute_stats(args):
    signature_path = OUTPUT / "assets/droid/source.json"
    stats_path = signature_path.with_name("norm_stats.json")
    if stats_path.is_file():
        if not signature_path.is_file() or json.loads(signature_path.read_text()) != stats_signature(args):
            raise ValueError("已有 DROID 统计与本次数据/参数不匹配；请保留原参数或移走输出目录后重新运行")
        print(f"复用 DROID 归一化统计：{stats_path}", flush=True)
        return
    import numpy as np
    from tqdm import tqdm

    from openpi.shared import normalize
    from openpi.training import data_loader

    config = make_config(args)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = raw_dataset(args, config, data_config, training=False)
    dataset = data_loader.IterableTransformedDataset(
        dataset, [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs], is_batched=True
    )
    statistics = {key: normalize.RunningStats() for key in ("state", "actions")}
    remaining = args.max_stats_frames
    batches = math.ceil(remaining / args.batch_size)
    for batch in tqdm(itertools.islice(dataset, batches), total=batches, desc="DROID normalization"):
        count = min(remaining, len(batch["state"]))
        for key, stats in statistics.items():
            stats.update(np.asarray(batch[key][:count]))
        remaining -= count
    if remaining:
        raise RuntimeError("DROID 数据迭代提前结束，未保存不完整统计")
    normalize.save(stats_path.parent, {key: stats.get_statistics() for key, stats in statistics.items()})
    write_json(signature_path, stats_signature(args))


class TorchRLDSLoader:
    """RLDS numpy batch -> 仓库 transforms -> torch tensor -> Observation。"""

    def __init__(self, args, config):
        from openpi.training import data_loader

        self.config = config.data.create(config.assets_dirs, config.model)
        dataset = raw_dataset(args, config, self.config, training=True)
        self.dataset = data_loader.transform_iterable_dataset(dataset, self.config, is_batched=True)

    def data_config(self):
        return self.config

    def __iter__(self):
        import jax
        import numpy as np
        import torch

        from openpi.models.model import Observation

        for batch in self.dataset:
            # 必须在 Observation.from_dict 之前转 torch，以启用它的 NHWC -> NCHW 转换。
            batch = jax.tree.map(lambda value: torch.from_numpy(np.array(value, copy=True)), batch)
            yield Observation.from_dict(batch), batch["actions"]


def train(args):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("全参数微调需要 NVIDIA CUDA GPU；本脚本不在 CPU/MPS 上启动训练")
    if not (BASE / "model.safetensors").is_file() or not (BASE / "source.json").is_file():
        raise FileNotFoundError("请先运行 --stage prepare")
    check_transformers()
    signature_path = OUTPUT / "assets/droid/source.json"
    if not signature_path.is_file() or json.loads(signature_path.read_text()) != stats_signature(args):
        raise ValueError("缺少匹配的 DROID 统计，请先使用相同参数运行 --stage stats")
    config = make_config(args)
    if config.checkpoint_dir.exists() and any(config.checkpoint_dir.iterdir()) and not args.resume:
        raise FileExistsError("训练目录已有文件；继续训练请加 --resume，另开实验请先移走 pi05_libero_droid")
    if args.resume and config.checkpoint_dir.exists():
        saved_steps = [
            int(path.name) for path in config.checkpoint_dir.iterdir() if path.is_dir() and path.name.isdigit()
        ]
        if saved_steps and max(saved_steps) > args.num_train_steps:
            raise ValueError("--num-train-steps 不能小于已有 checkpoint 的最新步数")
    # 只替换本进程的训练器回调，不改 scripts/train_pytorch.py 或 data_loader.py。
    spec = importlib.util.spec_from_file_location("openpi_libero_droid_trainer", ROOT / "scripts/train_pytorch.py")
    trainer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainer)
    trainer.build_datasets = lambda cfg: build_training_loader(args, cfg)
    original_save = trainer.save_checkpoint

    def save_checkpoint(model, optimizer, global_step, cfg, is_main, data_config):
        if global_step != cfg.num_train_steps and global_step % cfg.save_interval:
            return
        # 原训练器在递增 global_step 后保存，却用 num_train_steps-1 判断最后一步。
        # save_interval=1 仅用于已筛选的保存调用，保证任意总步数的最终模型都会保存。
        original_save(model, optimizer, global_step, dataclasses.replace(cfg, save_interval=1), is_main, data_config)

    trainer.save_checkpoint = save_checkpoint
    trainer.init_logging()
    trainer.train_loop(config)
    final = config.checkpoint_dir / str(args.num_train_steps)
    if not (final / "model.safetensors").is_file():
        raise FileNotFoundError(f"目标步数 checkpoint 不存在：{final}；检查 --resume 的目标总步数")
    # 硬链接避免复制数 GB 权重；跨文件系统时回退为复制，根目录可直接用于推理。
    temporary = OUTPUT / "model.safetensors.tmp"
    temporary.unlink(missing_ok=True)
    try:
        os.link(final / "model.safetensors", temporary)
    except OSError:
        shutil.copy2(final / "model.safetensors", temporary)
    temporary.replace(OUTPUT / "model.safetensors")
    shutil.copytree(final / "assets", OUTPUT / "assets", dirs_exist_ok=True)
    write_json(OUTPUT / "config.json", {
        "source_checkpoint": str(BASE),
        "source_uri": json.loads((BASE / "source.json").read_text())["source"],
        "model": dataclasses.asdict(config.model), "completed_steps": args.num_train_steps,
        "checkpoint": str(final.relative_to(OUTPUT)), "data": stats_signature(args),
        "batch_size": args.batch_size, "learning_rate": args.learning_rate,
        "inference_config": "finetune_pi05_libero_droid.make_config(args)",
    })
    print(f"训练完成：{OUTPUT / 'model.safetensors'}", flush=True)


def build_training_loader(args, config):
    loader = TorchRLDSLoader(args, config)
    return loader, loader.data_config()


def main():
    args = parse_args()
    if args.dry_run:
        print(json.dumps({"stage": args.stage, "source": args.source_checkpoint, "base": str(BASE),
                          "output": str(OUTPUT), "data": str(args.rlds_data_dir),
                          "download_droid": args.download_droid, "droid_max_gb": args.droid_max_gb,
                          "batch_size": args.batch_size,
                          "steps": args.num_train_steps, "resume": args.resume}, indent=2, ensure_ascii=False))
        return
    # JAX 只负责数据变换/权重转换，避免与 PyTorch 争用 CUDA 显存。
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.patch_transformers:
        patch_transformers()
    if args.stage == "all":
        # 阶段独立进程，释放转换器和 TensorFlow 统计进程的内存。
        for stage in ("prepare", "stats", "train"):
            command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--stage", stage]
            command = [item for item in command if item != "--patch-transformers"]
            if stage != "stats":
                command = [item for item in command if item != "--download-droid"]
            subprocess.run(command, cwd=ROOT, check=True)
    elif args.stage == "prepare":
        prepare_checkpoint(args)
    else:
        ensure_dataset(args)
        if args.stage == "stats":
            compute_stats(args)
        else:
            train(args)


if __name__ == "__main__":
    main()
