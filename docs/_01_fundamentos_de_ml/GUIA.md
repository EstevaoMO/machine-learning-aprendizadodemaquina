# Fundamentos de Aprendizado de Máquina

[← Anterior](../bem_vindo.md)

## Introdução
**Tom Mitchell**, em 1997, levantou uma definição formal para o conceito:
> "A computer program is said to learn from experience E for some class of tasks T and performance P, if its performance at tasks in T, as measured P, improves with experience E."

**Portanto, um programa de computacional aprende se, a partir de experiências, suas métricas de performance em uma tarefa específica melhoram.**

Imaginar isso em nível humano é relativamente simples. No entanto, refletir e estruturar esse "pensamento" para o computador é o desafio que o machine learning propõe-se a vencer. Esses algoritmos de **otimização** são fundamentados na inferência estatística para observação de padrões nos dados históricos de um determinado domínio (contexto). A partir desse aprendizado, o computador consegue, enfim, predizer o dado almejado; naturalmente, com uma margem de erro que vamos sempre tentar reduzir.

Esses modelos e algoritmos **podem** (e normalmente devem) **ser combinados para encontrar a saída desejada e com menor margem de erro**. Cada problema exigirá um conjunto de soluções específicas, e essa é essencialmente a habilidade que você desenvolverá ao longo deste curso. Você deverá ser capaz de, ao observar um determinado problema, determinar quais modelos e abordagens serão mais eficientes, aplicando algoritmos e extraindo informações estratégicas desses dados.

## Processo de Aprendizado
Assim como nós, seres humanos, aprendemos a partir das informações que adquirimos ao longo da vida, as máquinas aprendem a partir de **dados históricos**. Esses dados, por natureza, precisam ser de origem numérica; mais especificamente, expressos de **forma vetorial**. Do contrário, o computador não será capaz de "entender" o conteúdo ali descrito.

O processo lógico e matemático se difere da **análise prescritiva**, onde os dados são estudados e chega-se a uma conclusão dos eventos passados. O aprendizado de máquina nos permite alcançar a **análise preditiva**, onde desenvolvemos a habilidade de **inferir estatísticamente** os dados. Esse é um salto de análise que nos permite **antecipar diversos eventos**, como: ações no mercado financeiro, fenômenos metereológicos, preço de produtos, comportamentos de usuários.

> "A indução é a forma de inferência lógica que permite obter conclusões genéricas sobre um conjunto particular de exemplos. Ela é caracterizada como o raciocínio que se origina em um conceito específico e o generaliza, ou seja, da parte para o todo."
>
> [Conceitos sobre Aprendizado de Máquina, pg. 40 - MC Monard, JA Baranauskas (2003)](https://dcm.ffclrp.usp.br/~augusto/publications/2003-sistemas-inteligentes-cap4.pdf)

Essa natureza da indução de generalizar, no entanto, apresenta um problema que posteriormente descobriremos de forma prática como evitar em nosso modelo: o **viés** (bias).

Nossa amostra de dados, quando não bem selecionado e otimizada, pode nos gerar um **algoritmo falsamente preciso**. Ele pode estar muito bem treinado em uma categoria, mas **não apresentar uma boa capacidade de generalização para novos dados**. Por isso a importância do pré-processamento e tratamento de dados que veremos no próximo módulo.

> "Apesar da indução ser o recurso mais utilizado pelo cérebro humano para derivar conhecimento novo, ela deve ser utilizada com cautela, pois se o número de exemplos for insuficiente, ou se os exemplos não forem bem escolhidos, as hipóteses obtidas podem ser de pouco valor."
>
> [Conceitos sobre Aprendizado de Máquina, pg. 40 - MC Monard, JA Baranauskas (2003)](https://dcm.ffclrp.usp.br/~augusto/publications/2003-sistemas-inteligentes-cap4.pdf)

## Grandes Categorias de Problemas

*Não se preocupe em decorar todos os termos apresentados, apenas foque em absorver as ideias. Todos os termos mais técnicos serão explicados e aprofundados posteriormente.*

### **Aprendizado supervisionado**

É o paradigma dominante em aplicações comerciais na atualidade. A característica fundamental do aprendizado supervisionado é a presença da **variável-alvo**, também chamada de *target* ou *rótulo*. É chamado de "supervisionado" justamente porque exige essa variável previamente definida na base histórica para orientar e "supervisionar" as correções durante o treinamento do algoritmo.

A partir de um **conjunto de observações (*features*)**, infere-se uma generalização precisa em comparação à relação real, representada pela expressão $y \approx f(X)$. O algoritmo busca capturar e mapear essa relação de entrada-saída entre as variáveis explicativas e a resposta, permitindo realizar inferências estatísticas sólidas e prever novos cenários com dados inéditos.

**Principais Desafios:**
Apesar de poderoso, este modelo exige cuidado e validação. Requer monitoramento contínuo da capacidade de generalização do algoritmo, o que significa equilibrar a complexidade do modelo através do clássico **trade-off viés-variância**, evitando dois grandes problemas:

* **Underfitting (Subajuste):** Ocorre quando o modelo é simples demais para capturar a relação real dos dados.
* **Overfitting (Sobreajuste):** Ocorre quando o modelo se ajusta de forma excessiva e memoriza o ruído da base histórica, perdendo o poder de prever novos cenários.

Em geral, o *overfitting* será nosso maior desafio. Comumente os modelos tendem a "ficar viciados" nos padrões dos dados conhecidos, inferindo uma função $f(X)$ perfeita para dados similares aos dados de treinamento. Porém, quando novos dados chegam para esses modelos "viciados", eles não são capazes de prever corretamente o valor real. Veja na imagem a seguir:

*Imagem extraída de [https://aiml.com/what-are-some-options-to-address-overfitting-in-neural-networks/](https://aiml.com/what-are-some-options-to-address-overfitting-in-neural-networks/).*
<img src="https://external-content.duckduckgo.com/iu/?u=https%3A%2F%2Faiml.com%2Fwp-content%2Fuploads%2F2023%2F10%2Fproblem-of-overfitting_nn_mit.png&f=1&nofb=1&ipt=1e79315839f3f18ee6733de57ce4c4b284e194c70741a67a16f9859c5c6f6a3e" alt="Underffiting vs. Overfitting" />

Observe como no caso do underfitting, à esquerda, a função não descreve corretamente o comportamento dos dados; perceba como o erro médio entre o valor predito e o valor real é notório. Enquanto, à direita, a função descreve perfeitamente os dados treinados, no entanto, apenas os dados treinados. Qualquer dado novo, o modelo terá tanta dificuldade quanto o underfitting para generalizar o comportamento. Por isso, ao centro, temos a função ideal. A função acompanha a curva dos dados, sendo **capaz de generalizar para qualquer nova entrada**. Esse será sempre nosso objetivo.

Por fim, dentro do conceito de aprendizado supervisionado, temos duas principais classes de problemas:

* **Regressão**: É aplicada quando a variável-alvo é **quantitativa ou contínua**, assumindo valores numéricos reais. O objetivo é prever um valor numérico dentro de uma escala, como: preço de venda de uma casa, temperatura, duração, tamanho ou rendimento. Exemplos clássicos de algoritmos para esse fim incluem a regressão linear simples e múltipla.
* **Classificação**: É aplicada quando a variável-alvo é **qualitativa, categórica ou discreta**, representando grupos ou classes definidas. O objetivo é prever a qual categoria uma nova observação pertence, como: classificar se uma transação financeira é fraude, se um e-mail é spam ou se o mercado financeiro vai subir ou cair em determinado dia. Na prática, muitos classificadores calculam primeiro uma probabilidade contínua de pertencimento à classe (gerando um score entre 0 e 1) e depois aplicam um critério de corte ou limiar de decisão para fazer a classificação discreta final. Exemplos típicos incluem regressão logística, K-vizinhos mais próximos (KNN) e Máquinas de Vetores de Suporte (SVMs).

É fundamental entender que a grande maioria dos modelos **classificadores** não cospe a categoria final diretamente. Inicialmente, eles **estimam uma probabilidade contínua de sucesso** (um valor decimal entre 0 e 1). A classificação discreta final ocorre apenas quando aplicamos um **limiar** (threshold) ou critério de corte. Por exemplo, se determinarmos a regra de que uma probabilidade $p(X)≥0.5$ deve ser classificada como classe "positiva", apenas os dados que superarem essa métrica receberão o rótulo final. Essa métrica, naturalmente, surgirá de um processo analítico do domínio do problema.

### **Aprendizado não supervisionado**

É o paradigma exploratório por excelência em projetos de dados. A característica fundamental do aprendizado não supervisionado é a **ausência completa da variável-alvo** (rótulo). Neste cenário, o algoritmo não possui um "gabarito" prévio na base histórica para guiar e supervisionar o seu treinamento.

> "Algoritmos de aprendizado de máquina não supervisionado discernem padrões intrínsecos em dados não rotulados, como similaridades, correlações ou agrupamentos potenciais. Eles são mais úteis em cenários em que esses padrões não são necessariamente aparentes para observadores humanos. Como o aprendizado não supervisionado não pressupõe a preexistência de uma saída 'correta' conhecida, ele não exige sinais de supervisão ou funções de perda convencionais — portanto, 'não supervisionado'."
>
> [O que é aprendizado de máquina?, Dave Bergmann - IBM Think](https://www.ibm.com/br-pt/think/topics/machine-learning)


A partir de um conjunto contendo apenas variáveis explicativas ($X$), o modelo busca **inferir padrões ocultos, densidades ou correlações internas**. Em vez de mapear uma relação de entrada-saída explícita para prever um novo cenário, o objetivo é reestruturar ou simplificar a representação dos dados originais.

**Principais Desafios:**
Como não há uma "resposta correta" matemática, este modelo exige alto senso crítico e domínio do negócio. Requer validação humana da utilidade do modelo, evitando dois grandes obstáculos metodológicos:

* **Subjetividade:** Diferente dos modelos supervisionados, avaliar o sucesso aqui é mais desafiador. Não temos métricas absolutas (como precisão ou erro médio); o sucesso é validado pela utilidade prática da nova representação durante a análise exploratória.
* **Sensibilidade a ruídos e escalas:** Como o algoritmo tenta encontrar similaridades em tudo, variáveis irrelevantes ou em escalas muito discrepantes podem distorcer completamente os agrupamentos gerados, distorcendo-os no espaço vetorial. Veremos como tratar esse tipo de problema na etapa normalização dos dados.

Em geral, o desafio será dar significado ao que a máquina encontrou. Muitas vezes os algoritmos identificarão grupos matematicamente perfeitos, mas que não possuem lógica interpretável para o negócio. Veja na imagem a seguir:

*Imagem extraída de [https://www.mathworks.com/discovery/unsupervised-learning.html](https://www.mathworks.com/discovery/unsupervised-learning.html)*.

<img src="https://www.mathworks.com/discovery/unsupervised-learning/_jcr_content/mainParsys/band/mainParsys/lockedsubnav_copy/mainParsys/columns/335ce30a-77fd-4c27-af75-93672733f56b/columns/0e1a10a0-b289-4f36-9928-f1add97647b8/image_2128876021_cop.adapt.full.medium.png/1788512247796.png" alt="Aprendizado não supervisionado explicação gráfica"/>

Por fim, dentro do conceito de aprendizado não supervisionado, os mecanismos de sucesso dividem-se em duas principais ferramentas:

* **Agrupamento (clustering):** O objetivo é **separar os dados em subgrupos homogêneos** (clusters), onde os elementos de um mesmo grupo são estatisticamente mais semelhantes entre si do que em relação aos outros grupos. É amplamente aplicado em negócios para segmentação de mercado, identificação de perfis de consumo e sistemas de recomendação. Exemplos clássicos incluem K-Means e Agrupamento Hierárquico.
* **Redução de dimensionalidade:** O objetivo é **simplificar um conjunto de dados complexo, reduzindo o número de variáveis de entrada** ($X$) enquanto preserva a essência das informações originais (variância). É essencial para eliminar redundâncias e permitir a visualização de dados de alta dimensão. O exemplo mais notório é a Análise de Componentes Principais (PCA).

### **Aprendizado semi-supervisionado**
É uma abordagem híbrida crescente em aplicações modernas, posicionando-se estrategicamente entre os dois mundos anteriores. A característica fundamental do aprendizado semi-supervisionado é a utilização conjunta de um volume massivo de dados não rotulados combinado a uma pequena parcela de dados com a **variável-alvo** definida.

A partir desse conjunto misto de observações, o modelo tenta aprender a forma geral dos dados usando a porção não supervisionada e, em seguida, utiliza os poucos rótulos disponíveis para guiar e calibrar a predição.

**Principais Desafios:**
Apesar de eficiente, este modelo surge para resolver um problema de viabilidade financeira e operacional. O principal desafio que justifica e impulsiona o seu uso é o chamado **gargalo de rotulação de dados**:

* **Coleta vs. Rotulação:** Na realidade do mercado, coletar dados brutos ($X$) é um processo massivo e barato (como armazenar histórico de compras ou captar imagens). No entanto, rotular esses dados com o seu devido $y$ exige trabalho humano especializado e custoso. No cenário da detecção de fraudes, por exemplo, extrair dados de milhões de transações diárias é rotina, mas alocar auditores humanos para classificar manualmente cada uma delas para treinar um modelo supervisionado tradicional é operacionalmente inviável.

> "Rotular dados pode ser especialmente tedioso para determinados casos de uso. Em casos de uso mais especializados de aprendizado de máquina, como descoberta de medicamentos, sequenciamento genético ou classificação de proteínas, a anotação de dados não apenas é extremamente demorada, mas também requer expertise em domínio muito específico."
>
> [O que é aprendizado semissupervisionado?, Dave Bergmann - IBM Think](https://www.ibm.com/br-pt/think/topics/semi-supervised-learning)

Por fim, a principal abordagem utilizada dentro deste fluxo é:

* **A Metodologia do pseudo-rótulo (Pseudo-labeling):** O processo ocorre em etapas. Inicialmente, o algoritmo avalia a similaridade de todos os dados de forma não supervisionada. Em seguida, ele utiliza os poucos dados rotulados para "propagar" essa classificação e gerar classificações prévias (pseudo-rótulos) para os dados não rotulados adjacentes. Com a base agora "aumentada" e totalmente etiquetada, os dados alimentam um algoritmo supervisionado padrão para o treinamento final.

### **Aprendizado por reforço**

É um paradigma construído sobre um alicerce diferente dos anteriores, focado na tomada de decisão sequencial e na interação contínua. A característica fundamental do aprendizado por reforço é a figura de um **agente** que aprende através da **tentativa e erro** dentro de um **ambiente**.

> "O objetivo do agente é ganhar o máximo de recompensas possível ao longo do tempo. Ele faz isso aprendendo uma política, que é basicamente uma estratégia que diz qual ação tomar em qualquer situação. Essa política é refinada em várias iterações de interação com o ambiente."
>
> [O que é aprendizagem por reforço? - Google Cloud](https://cloud.google.com/discover/what-is-reinforcement-learning?hl=pt-BR)

O coração desta abordagem é o **ciclo interativo agente-ambiente**. A tomada de decisão ocorre de forma estritamente sequencial: cada ação escolhida e executada pelo agente altera o estado atual do próprio ambiente. Consequentemente, essa alteração influencia diretamente as recompensas (feedbacks positivos) ou punições (feedbacks negativos) que ele receberá nos passos futuros.

**Principais Desafios:**
Por ser um processo altamente dinâmico e interativo, o agente não recebe um "gabarito" apontando o caminho correto. No decorrer do treinamento, o algoritmo deverá gerenciar constantemente um conflito clássico da área, conhecido como o **dilema exploration vs. exploitation** (muito similar ao viés-variância):

* **Exploration (Exploração):** O agente tenta novas ações desconhecidas para mapear o ambiente e descobrir caminhos possivelmente melhores e mais eficientes a longo prazo.
* **Exploitation (Aproveitamento):** O agente utiliza as melhores ações já descobertas e consolidadas até o momento para garantir as maiores recompensas imediatas e seguras.

Para compreender esse *trade-off*, compare-o com a escolha de um restaurante para jantar. Você pode optar por ir sempre ao seu restaurante favorito de confiança, garantindo que a refeição será boa (*exploitation*). Por outro lado, você pode decidir testar um restaurante novo e desconhecido na cidade; correndo o risco de ter uma experiência ruim, mas com a chance de descobrir um lugar maravilhoso que se tornará o seu novo favorito (*exploration*). O aprendizado perfeito exige que a máquina saiba equilibrar ambas as atitudes.

A abordagem de aprendizado por reforço passa por conceitos mais profundos de machine learning, portanto, considere-se apenas apresentado ao conceito. Posteriormente, em tardes módulos, aprofundaremos esse conhecimento. No momento, não se desgaste tanto em compreendê-lo.

## Tipos de Variáveis e Predição

Embora a escolha da abordagem matemática do algoritmo seja definida exclusivamente pela natureza da variável-alvo ($Y$), as variáveis de entrada ($X$) podem ser uma mistura complexa de dados contínuos e categóricos. A única ressalva é que o computador precisa de representações numéricas; portanto, as variáveis categóricas devem ser devidamente transformadas e codificadas antes de entrarem na etapa de treinamento do modelo.

No mundo real, dados brutos raramente chegam perfeitamente estruturados e prontos para uso. Eles costumam apresentar ruídos, inconsistências e valores ausentes que exigem métodos rigorosos de **tratamento e imputação** para que o algoritmo não aprenda padrões errados.

Além disso, a grande maioria dos modelos matemáticos é altamente sensível à escala dos dados. Se estivermos lidando com variáveis contínuas de grandezas muito distintas no mesmo conjunto de entrada — como `idade` (variando de 18 a 80) e `renda anual` (variando na casa das dezenas de milhares) —, o modelo pode dar um peso falsamente maior à variável de maior magnitude simplesmente por seus números serem maiores.

Para corrigir essas distorções estatísticas, aplicamos técnicas de **escalonamento**, como a normalização e a padronização, garantindo que todas as características do conjunto de dados contribuam de forma equilibrada e justa para o aprendizado da máquina.

Todo esse processo de limpeza, transformação e adequação compõe uma etapa crítica chamada de **pré-processamento de dados**, e nós veremos isso com muito mais detalhes no próximo módulo.

## Veja Mais: IBM Technology

O vídeo a seguir deve ajudá-lo a entender melhor as diferenças entre os modelos e suas aplicações práticas.

<div class="video-wrapper">
  <iframe width="800" src="https://www.youtube.com/embed/W01tIRP_Rqs" title="Supervised vs. Unsupervised Learning" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" cc_load_policy=1 cc_lang_pref=pt allowfullscreen></iframe>
</div>

*Vídeo original por **IBM Technology** no [YouTube](https://youtu.be/W01tIRP_Rqs).*

---

[Próximo →](_02_pre_processamento_de_dados/GUIA.md)

---