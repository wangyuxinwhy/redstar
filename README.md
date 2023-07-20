# redstar

redstar 是开源透明的 LLM 评测工具

## 安装依赖

```bash
git clone https://github.com/wangyuxinwhy/redstar.git
pip install .
```

```bash
cd scripts
python eval.py --help
```

## 运行评测

```bash
python eval.py run --model azure_gpt_3_5 --task-name gsm8k_zero_shot
python eval.py list
```

```bash
python eval.py --model azure_gpt_3_5 --task-filter '"few_shot" in task.tags'
```
