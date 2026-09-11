# Módulo 2: Dados e Pré-processamento

[← Anterior](../_01_fundamentos_de_ml/GUIA.md)

## Introdução

No módulo anterior, estabelecemos que os algoritmos de aprendizado de máquina necessitam de dados históricos representados de forma vetorial e numérica. No entanto, **o mundo real é caótico.** Os dados que extraímos de bancos relacionais, APIs, planilhas ou sensores raramente chegam perfeitamente estruturados e prontos para alimentar um modelo matemático.

A etapa de pré-processamento muitas vezes consome a maior parte do tempo de um cientista de dados. É aqui que garantimos a integridade da informação. Modelos de machine learning são essencialmente motores matemáticos: se você os alimentar com "lixo" (dados ruidosos, ausentes ou em escalas distorcidas), eles fatalmente produzirão "lixo" em suas predições; princípio conhecido na computação como *Garbage In, Garbage Out* (GIGO).

Neste módulo, mergulharemos nas principais técnicas de engenharia e preparação de dados necessárias para criar modelos robustos e generalizáveis.

---

## Vazamento de Dados (Data Leakage)

Antes de aplicarmos qualquer técnica matemática de limpeza ou transformação que veremos a seguir, existe uma regra metodológica fundamental que não pode ser quebrada: **você deve separar e isolar o seu conjunto de teste imediatamente.**

O maior pecado que um cientista de dados pode cometer na etapa de pré-processamento chama-se **Vazamento de Dados** (*Data Leakage*). Esse fenômeno ocorre quando informações do conjunto de teste (os dados inéditos que usaremos para avaliar o modelo no futuro) contaminam o conjunto de treinamento (os dados que o modelo usa para aprender).

Quando isso ocorre, o modelo apresenta um **desempenho falso e excessivamente otimista** na fase de desenvolvimento, mas que sofre uma queda drástica de precisão ao ser colocado em produção. É o equivalente a entregar o gabarito da prova para um aluno antes de ele começar a estudar; a nota final dele no teste será máxima, mas a sua real capacidade de resolver problemas inéditos será zero.

**Como o vazamento acontece na prática?**

O vazamento geralmente não é intencional, ele é fruto de processos executados fora de ordem. As origens mais comuns são:

* **Pré-processamento Global:** Ocorre quando você calcula estatísticas (como a média para preencher dados nulos ou o valor máximo para escalonar variáveis) usando a base de dados inteira *antes* de separá-la. Se o algoritmo aprende a média global do arquivo, ele recebe conhecimento indireto sobre a distribuição dos dados de teste.
* **Vazamento Temporal (*Lookahead Bias*):** Em dados sequenciais (como o preço de ações ou histórico de vendas), ocorre quando a divisão entre treino e teste é feita de forma aleatória, ignorando a linha do tempo. Isso permite que o modelo utilize informações do "futuro" para prever eventos do "passado".
* **Registros Duplicados:** A presença de linhas idênticas espalhadas simultaneamente nos conjuntos de treino e teste faz com que o modelo apenas memorize as respostas em vez de aprender padrões matemáticos genéricos.

**Boas Práticas**

Para garantir que a estimativa de erro do seu modelo seja estatisticamente válida no mundo real, adote estas práticas como um mantra no seu fluxo de trabalho:

1. **Isole o conjunto de teste:** A primeira etapa prática do seu código deve ser a divisão dos dados. O conjunto de teste deve ser trancado em um "cofre" e permanecer intocado até o último segundo da avaliação final do sistema.
2. ***Fit* e *Transform*:** Todos os cálculos estatísticos (imputadores, escalonadores e codificadores) devem extrair seus parâmetros matemáticos **exclusivamente** dos dados de treinamento. Na sintaxe de código, isso significa usar a função de aprendizado (`fit`) apenas no treino, e a função de aplicação (`transform`) no treino e no teste.
3. **Respeite a cronologia:** Se o seu problema envolve séries temporais, as divisões não podem ser aleatórias. O conjunto de treinamento deve sempre anteceder cronologicamente os dados de validação.
4. **Encapsulamento (Pipelines):** A forma mais profissional e segura de prevenir vazamentos acidentais é combinar a preparação de dados e o modelo dentro de um *Pipeline*. Essa estrutura garante que o pré-processamento seja executado na ordem correta, recalculando as estatísticas do zero a cada etapa sem contaminar os ambientes.

---

## Tratamento de Dados Faltantes

O primeiro grande desafio ao recebermos um conjunto de dados é lidar com as **lacunas de informação**. Em bases reais, valores faltantes costumam vir representados por símbolos como `?`, `NA`, `null`, `NaN` ou mesmo espaços em branco.

É imperativo que você, como desenvolvedor, compreenda que um **valor nulo é conceitualmente diferente do número zero ou de um texto vazio**. Zero é uma informação quantitativa (ex: saldo bancário igual a zero); nulo representa a ausência total da informação (ex: não sabemos o saldo bancário). O problema é que a matemática falha ao tentar processar o "vazio".

> "A maioria dos algoritmos de Aprendizado de Máquina não consegue trabalhar com atributos faltantes [...]. Você tem três opções: 
> 
> 1. *Eliminar as instâncias correspondentes.*
> 2. *Eliminar o atributo inteiro.*
> 3. *Preencher os valores ausentes com algum valor* (zero, média, mediana etc.)."
>
> Traduzido de: [Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, pg. 63 - Aurélien Géron, O'Reilly®](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

Para lidar com essa ausência, aplicamos estratégias que variam desde intervenções simples até métodos estatísticos multivariados e iterativos:

* **Remoção de Registros (*Listwise Deletion*):** Consiste na eliminação completa das linhas que contêm valores ausentes. Esta é uma abordagem drástica e recomendada apenas quando o volume de dados apagados representa uma fração muito pequena da sua amostragem total. Perder dados significa reduzir o tamanho da amostra e comprometer o poder de generalização do modelo.
* **Imputação Estatística Simples (Univariada):** Consiste em substituir o dado faltante por uma medida de tendência central calculada sobre os dados observados. Para variáveis contínuas, frequentemente utilizamos a **média** ou a **mediana** (sendo a mediana mais segura caso existam valores extremos). Para variáveis categóricas, utilizamos a **moda**. Embora rápida, a imputação simples reduz artificialmente a variância dos dados e ignora a incerteza associada ao valor ausente.
* **Imputação Multivariada:** Utiliza as relações entre diferentes variáveis do conjunto de dados para estimar os valores ausentes. Em vez de imputar cada variável de forma independente, o método utiliza as demais variáveis disponíveis como preditoras, podendo realizar esse processo de forma iterativa. O `IterativeImputer` do *scikit-learn* implementa essa abordagem, estimando cada variável incompleta a partir das outras e repetindo o processo por múltiplas iterações.
* **Imputação Múltipla:** Consiste na criação de diferentes versões do conjunto de dados, preenchendo os valores ausentes com estimativas baseadas nas demais variáveis disponíveis. Em vez de atribuir um único valor ao dado faltante, o método gera múltiplas possibilidades, permitindo representar melhor a incerteza envolvida na imputação. É uma abordagem mais robusta que a imputação simples, especialmente quando há uma quantidade significativa de dados ausentes. O método MICE (*Multiple Imputation by Chained Equations*) será explicado em mais detalhes nos subtópicos a seguir.


Também, é fundamental **conhecer a ferramenta** que você está utilizando. Algoritmos baseados em **cálculo de distância geométrica** (como K-Vizinhos mais Próximos - KNN, Máquinas de Vetores de Suporte - SVM e PCA) **não possuem suporte nativo a dados faltantes** e falharão se não houver um tratamento prévio dos valores ausentes. 

Por outro lado, algoritmos modernos **baseados em árvores de decisão** e *boosting* (como o XGBoost) possuem recursos matemáticos integrados para **lidar diretamente com valores ausentes** durante o treinamento e teste, separando-os em ramificações específicas sem necessidade de imputação prévia.

### Mecanismos de Ausência

Para ir além das limitações da imputação simples, é necessário compreender o mecanismo que gerou a ausência nos dados. A teoria estatística divide a ausência em três categorias principais:

#### **MCAR** (*Missing Completely at Random*)
A probabilidade de um dado estar ausente é completamente independente tanto dos valores observados quanto dos que não foram observados. A ausência ocorre sem relação sistemática com as características dos dados.

<p align="center">$P(R \mid Y_{obs}, Y_{mis}) = P(R)$</p>

* **Exemplo:** Um sensor apresenta um mau funcionamento aleatório e deixa de registrar algumas medições, independentemente do valor que deveria ser registrado.

#### **MAR** (*Missing at Random*)
A probabilidade de um dado estar ausente depende de **outras variáveis observadas**, mas, uma vez consideradas essas variáveis, não depende do próprio valor que está ausente.

<p align="center">$P(R \mid Y_{obs}, Y_{mis}) = P(R \mid Y_{obs})$</p>

* **Exemplo:** A variável "Renda" possui valores ausentes com maior frequência em determinados grupos de idade. Se a idade está disponível na base, podemos utilizá-la para modelar e prever a distribuição dos valores ausentes de renda.

#### **MNAR** (*Missing Not at Random*)
A probabilidade de ausência depende do **próprio valor que está ausente**, mesmo após considerar as demais variáveis. A imputação tradicional aqui pode produzir estimativas enviesadas.

<p align="center">$P(R \mid Y_{obs}, Y_{mis}) \neq P(R \mid Y_{obs})$</p>

* **Exemplo:** Indivíduos com renda muito elevada têm maior probabilidade de não declarar sua renda justamente por ela ser alta. O próprio valor desconhecido influencia a ausência.

### Imputação Univariada vs. Multivariada

Quando decidimos preencher os dados faltantes matematicamente (cenários MCAR e MAR), podemos adotar duas abordagens principais em relação à dimensionalidade da análise:

* **Imputação Univariada:** Analisa de forma estritamente isolada apenas a coluna que possui o dado faltante. O exemplo clássico é substituir todos os nulos pela média ou mediana daquela mesma variável. Embora seja uma técnica rápida, ela ignora as relações com outras colunas e reduz artificialmente a variância global dos dados.
* **Imputação Multivariada:** Utiliza o conjunto de dados como um ecossistema. Em vez de olhar apenas para uma coluna, o algoritmo utiliza as **demais variáveis disponíveis como preditoras** para estimar o valor ausente. É uma técnica muito mais sofisticada e recomendada, pois preserva as correlações naturais e a variância do conjunto de dados.

### Imputação Múltipla e o Algoritmo MICE

O padrão-ouro no pré-processamento para lidar com dados faltantes de forma multivariada é a **Imputação Múltipla** (*Multiple Imputation*). 

Em abordagens simples (como preencher com a média), tapamos a lacuna com um único valor estático, o que gera uma falsa sensação de certeza para o modelo. A imputação múltipla corrige isso gerando $m$ valores plausíveis para cada dado faltante, produzindo $m$ versões diferentes e completas do seu conjunto de dados. Ao final do processo analítico, os resultados dessas múltiplas bases são combinados (*pooling*), refletindo matematicamente a verdadeira incerteza introduzida pelos dados ausentes.

Para realizar esse processo na prática, a ferramenta mais utilizada no mercado é o **MICE** (*Multivariate Imputation by Chained Equations*). 

**Qual a diferença entre eles?**

A *Imputação Múltipla* é a **metodologia estatística** conceitual de gerar e combinar várias bases para tratar a incerteza; o *MICE* é o **algoritmo prático** (código) utilizado para calcular e preencher essas bases.

**Como o MICE funciona?**

O algoritmo aplica o conceito de imputação multivariada de forma encadeada. Ele isola uma variável incompleta e utiliza todas as outras colunas da base para treinar um modelo de regressão e prever os espaços em branco. Em seguida, ele passa para a próxima variável incompleta e repete o processo. Esse ciclo é refeito por várias iterações (iterativo) até que as predições de todas as colunas se estabilizem, gerando bases imputadas altamente coerentes com a realidade.

Veja o vídeo a seguir para entender visualmente uma iteração do ciclo de imputação MICE.

<div class="video-wrapper">
  <iframe src="https://www.youtube.com/embed/zX-pacwVyvU" title="Multiple Imputation by Chained Equations (MICE)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</div>

*Vídeo original pelo usuário **philip9876** no [YouTube](https://youtu.be/zX-pacwVyvU).*

---

## Escalonamento e Padronização de Atributos

Uma vez que lidamos com as lacunas, precisamos olhar para as grandezas numéricas.

> "Uma das transformações mais importantes que você deve aplicar em seus dados chama-se feature scaling. Salvo alguns poucos casos, os algoritmos de Aprendizado de Máquina não apresentam bom desempenho quando os atributos numéricos de entrada possuem escalas muito diferentes."
>
> [Hands-on Machine Learning with Scikit-Learn, Keras & TensorFlow, pg. 69 - Aurélien Géron, O'Reilly (2019)](https://www.oreilly.com/library/view/hands-on-machine-learning/9781492032632/)

Imagine que estamos prestando prever a aprovação de crédito usando duas variáveis preditoras: a `idade` (variando de 18 a 80 anos) e a `renda anual` (variando de 15.000 a 200.000 reais).

> "Dado que o classificador KNN prevê a classe identificando as observações mais próximas, a escala das variáveis importa. Qualquer variável em grande escala terá um impacto muito maior na distância."
>
> [An Introduction to Statistical Learning, pg. 165 - Gareth James et al., Springer (2013)](https://www.statlearning.com/)

Nesse cenário, **algoritmos que calculam distâncias geométricas sofrerão distorções severas**. A variável de renda anual dominará artificialmente a métrica de distância devido à sua amplitude; o algoritmo considerará erroneamente que uma diferença de 1.000 reais é infinitamente mais importante do que 20 anos de vida.

Para evitar isso, aplicamos técnicas de transformação para encaixar os dados em escalas amigáveis. Assim, os modelos convergem com mais facilidade. 

*Imagem extraída de [GeeksforGeeks](https://www.geeksforgeeks.org/machine-learning/standardscaler-minmaxscaler-and-robustscaler-techniques-ml/).*

<img width=1000 src="https://media.geeksforgeeks.org/wp-content/uploads/20250418115633045690/scaling-o.webp" alt="Gráfico das diferenças dos métodos de escalonamento e padronização"/>

### Padronização Z-score (Standardization)

Técnica de escalonamento mais utilizada na indústria. O seu objetivo é centralizar os dados, forçando-os a ter uma **média aritmética igual a zero** ($\mu = 0$) e um **desvio padrão igual a um** ($\sigma = 1$). 

Geometricamente, essa transformação desloca a distribuição para o centro do eixo coordenado, mas preserva a forma original da curva. É a escolha ideal para algoritmos que assumem que os dados de entrada seguem uma distribuição Gaussiana.

$$x_{scaled} = \frac{x - \mu}{\sigma}$$

*Nota: A padronização não impõe limite rígido aos valores máximos e mínimos. Extremos continuarão existindo representados como desvios padrão distantes da média.*

### Escalonamento Min-Max (Normalization)

Não se preocupa com a média ou com o desvio padrão. O objetivo do `MinMaxScaler` é espremer todos os dados para que caibam dentro de um intervalo fixo, que por padrão é **entre 0 e 1**. O menor valor observado recebe o valor $0$ e o maior recebe o valor $1$.

$$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

*Atenção: Altamente sensível a outliers. Se houver um erro de digitação na base (ex: idade 999), ele será o valor 1.0, e todas as idades reais serão espremidas próximas ao 0.*

### Escalonamento pelo Valor Absoluto Máximo

O `MaxAbsScaler` apenas divide cada número pelo maior valor absoluto encontrado na coluna, contendo os dados no intervalo de **[-1, 1]**.

A grande vantagem de não realizar subtrações é que **ele não destrói a esparsidade dos dados**. Um valor zero continuará sendo zero após a transformação; preservando a integridade de matrizes muito esparsas.

$$x_{scaled} = \frac{x}{\max(\vert{}x\vert{})}$$

### Escalonamento Robusto

Técnicas que dependem da média ou do valor máximo falham quando a base possui valores extremos. O `RobustScaler` subtrai a **mediana** ($Q_2$), independentemente dos extremos, e divide pelo **intervalo interquartil (IQR)**. A massa principal dos dados é escalonada corretamente e os outliers são isolados.

$$x_{scaled} = \frac{x - Q_2}{Q_3 - Q_1}$$

Para entender visualmente as diferenças geométricas entre normalização e padronização, confira:

<div class="video-wrapper">
  <iframe width="800" src="https://www.youtube.com/embed/sxEqtjLC0aM" title="Standardization vs Normalization Clearly Explained!" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>
</div>

*Vídeo original por **Normalized Nerd** no [YouTube](https://www.youtube.com/sxEqtjLC0aM).*

## Codificação de Variáveis Categóricas

Como os modelos processam apenas números, precisamos traduzir **variáveis qualitativas** para representações numéricas. Elas dividem-se em dois tipos: 

* **Ordinais:** Possuem hierarquia lógica (ex: escolaridade ou avaliações "ruim, médio, bom").
* **Nominais:** Não possuem relação de ordem (ex: cores ou nomes de países).

**Como transformá-las?**

* **Codificação ordinal:** Converte cada categoria em um valor inteiro sequencial (ruim = 1, médio = 2, bom = 3).
* **One-hot encoding:** Cria uma nova coluna binária (0 ou 1) para cada categoria possível da variável. 

Precisamos evitar a armadilha das variáveis *dummy* (a multicolinearidade perfeita). Se uma variável possui $C$ categorias, a soma de suas colunas binárias será sempre previsível, quebrando cálculos de modelos lineares. Por isso, **removemos sempre uma categoria**, utilizando $C-1$ colunas; a categoria omitida torna-se a *baseline* de comparação do modelo.

Considere a codificação do atributo `sexo`, assumindo os valores `masculino` ou `feminino`:

| Pessoa | Masculino | Feminino |
| :--- | :---: | :---: |
| A      |         1 |        0 |
| B      |         0 |        1 |
| C      |         1 |        0 |

Perceba que a soma das colunas assumirá sempre o valor $1$. A codificação correta com o drop de uma coluna (neste caso, a feminina) fica assim:

| Pessoa | Masculino |
| :--- | :---: |
| A      |         1 |
| B      |         0 |
| C      |         1 |

O valor zero agora nos diz tacitamente que o indivíduo pertence à classe removida. Essa abordagem se aplica a qualquer cardinalidade no *one-hot encoding*.

Quando lidamos com alta cardinalidade (uma variável com milhares de classes únicas), o processo criaria milhares de colunas, explodindo a dimensionalidade. Nesses casos, aplicamos o **agrupamento de raras**, fundindo categorias de baixíssima frequência em uma única classe "Outros"; ou codificações avançadas como o *target encoding*.

## Detecção e Tratamento de Outliers e Ruídos

O último pilar do pré-processamento é garantir que pontos excepcionalmente bizarros não distorçam a visão do modelo. 

**Outliers são observações que se desviam drasticamente do padrão natural da amostra**. Podem surgir por falhas de medição, erros humanos ou eventos legítimos raros. O impacto é severo; inflam métricas de erro ($RMSE$) e reduzem a capacidade preditiva em algoritmos sensíveis à variância.

Para análise, atentamo-nos ao **Intervalo Interquartil (IQR)**. Consideramos discrepante qualquer valor abaixo do limite inferior ($Q_1 - 1.5 \times IQR$) ou acima do superior ($Q_3 + 1.5 \times IQR$). Gráficos de *boxplot* são excelentes ferramentas visuais para essa detecção.

*Imagem extraída de [https://fernandafperes.com.br/blog/interpretacao-boxplot/](https://fernandafperes.com.br/blog/interpretacao-boxplot/).*

<img width=600 src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Ffernandafperes.com.br%2Fblog%2Finterpretacao-boxplot%2Fg2.png&f=1&nofb=1&ipt=f189d1d4cff722cdf9ec3a98566e9959e84ae3c4047244dc2fefa2689cd310a5" alt="Explicação visual do gráfico boxplot"/>

**Estratégias de Mitigação (Tratamento):**

* **Remoção justificada:** Exclusão da linha apenas quando for inequivocamente comprovado que a observação é erro de coleta (ex: pessoa com 999 anos).
* **Truncamento (*winsorization*):** Limitação estatística fixando valores extremos a um teto ou piso. Exemplo: qualquer idade acima de 90 anos passa a ser registrada fixamente como 90.
* **Transformações:** Aplicação de logaritmo para comprimir a escala dos dados. Reduz o impacto matemático de grandes valores sem removê-los, útil em variáveis com forte assimetria (renda, faturamento).

---

[Próximo →](../_03_avaliacao_generalizacao/GUIA.md)