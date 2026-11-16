# 数据与第三方软件来源

教学代码采用仓库根目录的 MIT 许可证。该许可证不替代数据提供方或外部模型的条款。

示例数据来自 NASA/JPL 的 GRACE-FO Level-1B RL04 CLK1B 日产品，卫星 C、D，名义日期为 2025-06-01 至 2025-06-03。原始文件含跨日重叠段。项目使用其 `eps_time`，没有使用轨道数据。

- 原始文件头产品标识：[10.5067/GFJPL-L1B04](https://doi.org/10.5067/GFJPL-L1B04)。
- PO.DAAC ASCII 集合：[GRACEFO_L1B_ASCII_GRAV_JPL_RL04](https://podaac.jpl.nasa.gov/dataset/GRACEFO_L1B_ASCII_GRAV_JPL_RL04)，集合 DOI：[10.5067/GFL1B-ASJ04](https://doi.org/10.5067/GFL1B-ASJ04)。两个 DOI 的标识层次不同，均按来源记录。
- [官方用户手册](https://isdc-data.gfz.de/grace-fo/DOCUMENTS/Level-1/GRACE-FO_L1_Data_Product_User_Handbook_20190911.pdf)，CLK1B 字段和质量标记见 4.2.4。
- 原始文件头声明的数据使用政策：[NASA Earth Science Data Policy](https://science.nasa.gov/earth-science/earth-science-data/data-information-policy)。
- [PO.DAAC 引用指南](https://podaac.jpl.nasa.gov/CitingPODAAC)。使用数据时应致谢 NASA/JPL、GRACE-FO 科学数据系统及 NASA/GFZ 联合任务。

`data/sample/*.txt` 是本仓库处理得到的教学子集，不是官方原始文件。相邻 manifest 记录六份原始日文件的文件名和 SHA-256、处理规则、行数、输出哈希；记录中不包含本机路径。处理规则与再生方法见 [数据说明](advanced/docs/data.md)。

TimesFM 通过独立依赖安装，仓库不分发其源码或权重。其[源码](https://github.com/google-research/timesfm)使用 Apache-2.0；3.0 [权重许可证](https://huggingface.co/google/timesfm-3.0-pytorch/blob/main/LICENSE)限制下载权重的商业和生产用途。本项目的可选路径用于课堂和本地研究。CPU 多项式基线不依赖 TimesFM。

简明版 `data/clock.csv` 从上述 C 星子集的全部记录提取时间与钟差两列，支持固定 8 个起报窗口；默认窗口 1 保留原演示时间，详情见 [说明](data/README.md)。
