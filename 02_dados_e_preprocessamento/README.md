**[← Página inicial](../README.md)** · **[Guia do módulo →](../docs/_02_dados_e_preprocessamento/GUIA.md)**

[Módulo 03: Avaliação de Modelos e Generalização →](../03_avaliacao_generalizacao/README.md)

---

# Dados e Pré-processamento

> Técnicas de limpeza, transformação e adequação de dados para garantir a integridade da informação antes da modelagem matemática.

Este módulo aborda a etapa mais crítica e demorada do ciclo de vida de um projeto de Machine Learning. O objetivo é compreender como lidar com as imperfeições do mundo real e transformar dados brutos e caóticos em uma base refinada e de alta precisão preditiva, evitando o princípio estrutural de *Garbage In, Garbage Out* (GIGO).

## 🎯 Objetivos

Ao concluir este módulo, você deverá ser capaz de:

- Entender e prevenir o Vazamento de Dados (*Data Leakage*) garantindo a correta separação entre treino e teste.
- Diagnosticar mecanismos de ausência e aplicar técnicas univariadas ou multivariadas (como MICE) para o tratamento de dados faltantes.
- Diferenciar e aplicar técnicas de escalonamento (Padronização e Normalização) para adequar grandezas numéricas sensíveis.
- Transformar variáveis qualitativas em numéricas através de codificações ordinais e *one-hot encoding*, contornando a multicolinearidade.
- Detectar anomalias (*outliers*) e adotar estratégias analíticas e estatísticas para mitigar seus impactos nos modelos.

## 📋 Conteúdo

1. [Vazamento de Dados (Data Leakage)](https://estevaomo.github.io/machine-learning-aprendizadodemaquina/_02_dados_e_preprocessamento/GUIA/#vazamento-de-dados-data-leakage)
2. [Tratamento de Dados Faltantes](https://estevaomo.github.io/machine-learning-aprendizadodemaquina/_02_dados_e_preprocessamento/GUIA/#tratamento-de-dados-faltantes)
3. [Escalonamento e Padronização de Atributos](https://estevaomo.github.io/machine-learning-aprendizadodemaquina/_02_dados_e_preprocessamento/GUIA/#escalonamento-e-padronizacao-de-atributos)
4. [Codificação de Variáveis Categóricas](https://estevaomo.github.io/machine-learning-aprendizadodemaquina/_02_dados_e_preprocessamento/GUIA/#codificacao-de-variaveis-categoricas)
5. [Detecção e Tratamento de Outliers e Ruídos](https://estevaomo.github.io/machine-learning-aprendizadodemaquina/_02_dados_e_preprocessamento/GUIA/#deteccao-e-tratamento-de-outliers-e-ruidos)

## 🧪 Laboratórios

Os conceitos deste módulo são explorados nos notebooks disponíveis em:

[📓 Mapa de Colabs](./colabs/MAPA.md)

A prática deste módulo possui caráter técnico e ferramental, priorizando a **manipulação de conjuntos de dados reais, uso de bibliotecas como Scikit-Learn e a construção de *pipelines*** para garantir um pré-processamento limpo e totalmente livre de vazamentos.

## 📖 Materiais de Referência

- [Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow; Aurélien Géron (2019)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)
- [An Introduction to Statistical Learning; Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani (2013, 2021)](https://www.statlearning.com/)
- [StandardScaler, MinMaxScaler and RobustScaler Techniques - GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/)
- [Interpretação do Boxplot - Fernanda F. Peres](https://fernandafperes.com.br/blog/interpretacao-boxplot/)
- [Multiple Imputation by Chained Equations (MICE) - philip9876 (YouTube)](https://youtu.be/zX-pacwVyvU)
- [Standardization vs Normalization Clearly Explained - Normalized Nerd (YouTube)](https://www.youtube.com/sxEqtjLC0aM)

---

[Módulo 03: Avaliação de Modelos e Generalização →](../03_avaliacao_generalizacao/README.md)