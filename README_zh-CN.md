<p align="center">
  <img src="figures/Cobra.png" alt="CoBRA Logo" width="400"/>
</p>

# CoBRA: Cognitive Bias Regulator for Social Agents

<p align="center">
  <a href="https://arxiv.org/abs/2509.13588"><img src="https://img.shields.io/badge/arXiv-2509.13588-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.48550/arXiv.2509.13588"><img src="https://img.shields.io/badge/DOI-10.48550/arXiv.2509.13588-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>

**为基于LLM的社交模拟提供可编程的认知偏差控制**

> 📄 **论文**: [arXiv:2509.13588v2](https://arxiv.org/abs/2509.13588) - *Programmable Cognitive Bias in Social Agents*

**📖 Language / 语言**: [English](README.md) | [繁體中文](README_zh-TW.md)

**CoBRA** (Cognitive Bias Regulator for Social Agents / 社交代理的认知偏差调节器) 是一个用于大语言模型(LLM)中认知偏差可控调节的通用框架。它使用 **表示工程(RepE)** 和 **提示工程(Prompt Engineering)** 来精确控制AI系统中的偏差行为。

## CoBRA 是什么?

CoBRA 提供了一个**统一框架**用于:
- 🎯 **精确控制** LLM中的4种关键认知偏差(权威偏差、从众偏差、确认偏差、框架效应)
- 🧠 **表示工程(RepE)**: 通过操纵模型激活值来实现细粒度偏差控制
- 💬 **提示工程**: 使用Likert量表提示进行基准对比
- 📊 **可重现实验**: 包含完整的代码、数据和分析工具

## 视觉概览

![图1: CoBRA框架概览](figures/fig1.png)
*图1: 概览图展示了CoBRA框架如何调节LLM中的偏差。我们首先从涉及特定认知偏差的对话中提取代表对照组(无偏差)和实验组(有偏差)的文本对。然后,我们利用这些文本对生成正负样本对,以训练偏差方向。在推理阶段,我们展示如何操纵模型的隐藏表示以实现对偏差程度的精细控制。*

![图2: 用Likert量表制定提示的说明](figures/fig2.png)
*图2: 用于提示工程基准的Likert量表提示构建。(a) 权威提示的高级直觉和(b) 完整提示示例,包括详细的5点Likert量表定义和任务特定指令。*

<details>
<summary><b>📊 点击查看技术细节图表(图5-6)</b></summary>

![图5: RepE和提示方法的层级消融结果](figures/fig5.png)
*图5: RepE控制在不同Transformer层级的有效性。结果来自于Mistral-7B-Instruct-v0.3模型在6个不同社会心理学场景上的测试,每个场景涵盖4种偏差类型。图表展示了Y轴表示的偏差行为相对于未调节基线的变化。RepE干预主要影响中间层(12-22层),符合先前研究关于概念信息定位的发现。有趣的是,浅层和深层的干预产生了最小的行为改变,这表明关键的偏差相关表示主要编码在中间层级。*

![图6: 多样化角色在多项选择偏差测量中的消融研究](figures/fig6.png)
*图6: 在25个不同角色上的偏差控制有效性。我们在Asch从众实验(上)和Milgram服从实验(下)中评估CoBRA,使用了从"亚当·斯密"(经济学家)到"沃尔夫冈·舒尔茨"(化学学生)等25个不同背景的角色。每个子图都显示了各个角色在5个Likert偏差水平下的平均选择概率,展示了一致的偏差控制效果,无论角色背景如何。*

</details>

---

## 快速开始

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/CoBRA.git
cd CoBRA
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 运行实验
```bash
# 对权威偏差运行RepE实验
python examples/unified_bias/pipelines.py --bias authority --method repe

# 对确认偏差运行提示基准实验
python examples/unified_bias/pipelines.py --bias confirmation --method prompt_likert

# 对所有偏差运行完整批量实验
python examples/unified_bias/run_batch.py
```

查看 [统一偏差实验README](examples/unified_bias/README.md) 了解完整的使用指南。

---

## 仓库结构

```
CoBRA/
├── control/              # 核心RepE和提示引擎
│   ├── repe_experiment.py
│   ├── prompt_experiment.py
│   └── base.py
├── data/                 # 偏差场景和提示(4种偏差类型)
│   ├── authority/
│   ├── bandwagon/
│   ├── confirmation/
│   └── framing/
├── examples/             # 可重现实验
│   └── unified_bias/     # 统一实验框架 📖 [README](examples/unified_bias/README.md)
│       ├── pipelines.py
│       ├── run_batch.py
│       └── ablation/     # 消融研究 📖 [README](examples/unified_bias/ablation/README.md)
│           └── api_experiments/ # 封闭源码模型实验 📖 [README](examples/unified_bias/ablation/api_experiments/README.md)
├── generator/            # 场景生成工具 📖 [README](generator/README.md)
├── demo/                 # Facebook从众情绪实验 📖 [README](demo/README.md)
├── webdemo/              # Web界面演示 📖 [README](webdemo/README.md)
└── figures/              # 论文图表 📖 [README](figures/README.md)
```

---

## 关键组件

| 组件 | 描述 | 文档 |
|------|------|------|
| **Control** | RepE和提示工程实验的核心引擎 | [control/](control/) |
| **Unified Bias** | 统一的偏差实验框架(推荐用于可重现性) | [examples/unified_bias/README.md](examples/unified_bias/README.md) |
| **Ablation** | 角色、温度和模型的消融研究 | [examples/unified_bias/ablation/README.md](examples/unified_bias/ablation/README.md) |
| **Generator** | OpenRouter驱动的场景生成 | [generator/README.md](generator/README.md) |
| **Demo** | 完整的Facebook从众偏差演示 | [demo/README.md](demo/README.md) |

---

## 支持的偏差类型

| 偏差类型 | 描述 | 数据目录 |
|---------|------|---------|
| **Authority (权威偏差)** | 倾向于服从权威人物 | [data/authority/](data/authority/) |
| **Bandwagon (从众偏差)** | 倾向于跟随群体意见 | [data/bandwagon/](data/bandwagon/) |
| **Confirmation (确认偏差)** | 倾向于寻找确认既有信念的信息 | [data/confirmation/](data/confirmation/) |
| **Framing (框架效应)** | 决策受问题表达方式影响 | [data/framing/](data/framing/) |

---

## 引用

如果您在研究中使用CoBRA,请引用我们的论文:

```bibtex
@misc{yao2025cobra,
  title={CoBRA: Cognitive Bias Representation and Adjustment in Large Language Models},
  author={Yao, Boshi and Wang, Kefan and Hu, Yixin and Zhang, Yang and Zhou, Qiyang and Wang, Tong and Wang, Yujia and Bhattamishra, Sadhika and Fung, Yi},
  year={2025},
  eprint={2509.13588},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2509.13588},
  doi={10.48550/arXiv.2509.13588}
}
```

**预印本链接**: [https://arxiv.org/abs/2509.13588](https://arxiv.org/abs/2509.13588)  
**DOI**: [https://doi.org/10.48550/arXiv.2509.13588](https://doi.org/10.48550/arXiv.2509.13588)

---

## 许可证

本项目采用Apache 2.0许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 致谢

我们感谢开源社区对本项目的支持,以及所有提供反馈和贡献的研究人员。

---

## 联系方式

如有问题或合作咨询,请通过以下方式联系:
- 📧 Email: [您的邮箱]
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/CoBRA/issues)
