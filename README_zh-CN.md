<p align="center">
  <img src="figures/Cobra.png" alt="CoBRA Logo" width="400"/>
</p>

# CoBRA: 实现跨模型精确一致的智能体行为

<p align="center">
  <a href="https://arxiv.org/abs/2509.13588"><img src="https://img.shields.io/badge/arXiv-2509.13588-b31b1b.svg" alt="arXiv"></a>
  <a href="https://doi.org/10.48550/arXiv.2509.13588"><img src="https://img.shields.io/badge/DOI-10.48550/arXiv.2509.13588-blue" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License"></a>
</p>


> 📄 **论文**: [arXiv:2509.13588v2](https://arxiv.org/abs/2509.13588) - *Programmable Cognitive Bias in Social Agents*

**📖 Language / 语言**: [English](README.md) | [繁體中文](README_zh-TW.md)

**CoBRA (Cognitive Bias Regulator for Social Agents / 社交代理的认知偏差调节器)** 利用结构化且经过验证的心理学实验作为校准工具包来控制和对齐跨模型的模型行为。

## 问题与解决方案

<p align="center">
  <img src="figures/fig1.png" alt="CoBRA Overview" width="800"/>
  <br>
</p>

现有的社交模拟实验通常使用隐式的自然语言描述来指定智能体行为。然而，我们发现这些规范往往导致不一致和不可预测的智能体行为。例如，(A) 现实世界中的经济学家应该比普通大众更不容易受到框架效应的影响；(B) 然而，基于隐式自然语言规范的智能体经常在不同模型中产生不一致的行为，角色间预期的行为差异无法可靠观察到。

(C) 为了解决这一挑战，我们引入了 **CoBRA**，它使研究人员能够定量地明确指定基于LLM的智能体的认知偏差，从而在跨模型中产生精确和一致的行为。

---

**CoBRA 提供三种控制方法:**
- **提示工程** (输入空间控制)
- **表示工程** (激活空间控制)  
- **微调** (参数空间控制)

以下是CoBRA的闭环工作流程示例。一位社会科学家旨在创建具有中等框架效应的智能体（例如，在0-4量表上得分2.6）。(1) 她在CoBRA中指定所需的偏差水平，同时提供自然语言智能体描述。(2) CoBRA使用经过验证的经典社会科学实验（例如，亚洲疾病研究）测量智能体的框架效应。(3) 如果测量的偏差偏离规范，行为调节引擎通过提示工程、激活修改或微调迭代调整智能体，直到智能体可靠地表现出目标偏差。

<p align="center">
  <img src="figures/fig2.png" alt="CoBRA Workflow" width="800"/>
  <br>
</p>




<details>
<summary><b>点击查看更多技术图表</b></summary>

<p align="center">
  <img src="figures/fig5.png" alt="Classic Social Experiment Testbed" width="800"/>
  <br>
  <em>图5: 经典社会实验测试平台。结构化知识库由编码行为模式及其相应的经典社会实验范式组成。智能体暴露在基于场景的经典社会实验中，这些实验旨在引发特定类型的认知偏差。这些场景使用可调整占位符的提示模板构建，智能体响应通过李克特量表收集。基于这些响应，计算认知偏差指数来量化智能体行为。</em>
</p>

<p align="center">
  <img src="figures/fig6.png" alt="Behavioral Regulation Engine" width="800"/>
  <br>
  <em>图6: 行为调节引擎。引擎提供三种控制方法，涵盖基于LLM的智能体的所有可能干预空间：输入空间的提示工程、激活（隐藏状态）空间的表示工程和参数空间的微调。所有方法都与经典社会实验测试平台集成，并使用相应的控制系数来校准认知偏差指数。</em>
</p>

</details>

## 快速开始（3步）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 进入统一偏差控制模块
cd examples/unified_bias

# 3. 运行偏差实验
python pipelines.py --bias authority --method repe-linear --model Mistral-7B
```

**就这样。** 系统将测量和控制智能体的权威效应偏差。

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
