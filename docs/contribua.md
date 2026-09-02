# Contribua | Manual Técnico

Este é um projeto **comunitário e colaborativo**, criado para reunir materiais gratuitos de estudo sobre Machine Learning em português.

Estudantes, professores e profissionais são convidados a contribuir, independentemente do nível de experiência. Uma contribuição não precisa ser grande: uma correção, uma referência, uma explicação melhor ou um novo experimento já pode ajudar outras pessoas a aprender.

> **Para colaborar com novos tópicos e implementações, consulte a [ementa do curso](./index.md) e veja o que ainda falta para ser feito.**

## 1. **Sobre o projeto**

O projeto está em constante construção. Correções de erros conceituais, fórmulas incorretas, bugs, melhorias de implementação, novos algoritmos, projetos e refinamentos dos materiais são extremamente bem-vindos.

A ideia é construir coletivamente um material de estudo **aberto, acessível e tecnicamente fundamentado**.

Contribua com aquilo que você sabe e, principalmente, esteja aberto a aprender com as contribuições de outras pessoas.

---

## 2. **Boas práticas**

### Referências e licenças

Ao utilizar materiais externos:

* Verifique sua licença de uso antes de reproduzir ou adaptar conteúdo.
* Dê os devidos créditos aos autores.
* Referencie livros, artigos, cursos, documentações e outras fontes utilizadas.
* Não copie materiais protegidos por direitos autorais sem a devida autorização.
* Prefira fontes consolidadas e bem fundamentadas.

Explicações e resoluções devem, sempre que possível, ser fundamentadas em materiais reconhecidos na academia ou na comunidade técnica.

### 3. *Qualidade do conteúdo**

Evite produzir conteúdo excessivamente genérico ou superficial.

Ao criar uma implementação ou projeto, procure compreender:

* qual problema está sendo resolvido;
* qual é o objetivo da implementação;
* por que determinado algoritmo ou modelo foi escolhido;
* quais são suas hipóteses e limitações;
* como os resultados devem ser interpretados.

O objetivo é **entender e explicar o que está sendo feito**.

### 4. **Uso de Inteligência Artificial**

Ferramentas de Inteligência Artificial podem auxiliar no desenvolvimento das contribuições, especialmente na escrita, revisão, organização e programação.

Entretanto, seu uso deve ser consciente.

Não envie código ou textos gerados por IA sem compreender, revisar e validar o resultado.

Ao utilizar IA:

* analise o código antes de submetê-lo;
* verifique se as explicações estão corretas;
* valide fórmulas e resultados;
* confira todas as referências;
* nunca invente ou aceite referências bibliográficas sem verificar sua existência;
* utilize os materiais já existentes no projeto como referência de estrutura e estilo.

A IA deve **auxiliar o aprendizado, não substituí-lo**.

Nem todos possuem facilidade para escrever textos com excelência. Ferramentas de IA podem ajudar nesse processo, mas contribuições genéricas ou pouco fundamentadas dificilmente serão incorporadas ao projeto sem revisão.

---

## 5. **Branches**

A organização de branches deve permanecer simples.

### `main`

A branch `main` representa a versão estável e publicável do projeto.

### Branches de contribuição

Para novas contribuições, crie uma branch específica:

```text
novo/nome-da-contribuicao
corr/nome-da-correcao
docs/nome-da-documentacao
```

Exemplos:

```text
novo/regressao-linear
novo/colab-kmeans
corr/formula-pca
docs/melhorar-guia-regressao
```

Após finalizar a contribuição, abra um **Pull Request** para `main`.

Branches temporárias devem ser removidas após o merge, quando não forem mais necessárias.

---

## 6. **Commits**

Os commits devem seguir uma estrutura consistente para facilitar o entendimento do histórico, o versionamento e a identificação de alterações.

### **Formato**

```text
<tipo>(<escopo>): <descrição>
```

As descrições devem ser objetivas e, preferencialmente, seguir o formato:

```text
verbo no gerúndio + objeto da ação + finalidade (opcional)
```

Exemplo:

```text
novo(regressão): adicionando projeto de predição binária
```

A descrição não precisa conter todas as informações sobre a alteração. Use apenas as palavras necessárias para comunicar claramente o que foi feito.

### Tipos

| Tipo       | Uso                                       |
| ---------- | ----------------------------------------- |
| `novo`     | Novo conteúdo, notebook ou funcionalidade |
| `docs`     | Alterações na documentação                |
| `corr`     | Correções de erros ou problemas           |
| `refat`    | Reorganização ou melhoria estrutural      |
| `config`   | Alterações de configuração e manutenção   |

### **Escopo**

Não existem palavras-chave obrigatórias para o escopo.

Utilize nomes que façam sentido para a alteração, como:

```text
regressao
classificacao
pca
kmeans
guias
colabs
readme
```

Observe commits anteriores antes de escolher um novo termo. Quando já existir um padrão estabelecido, procure mantê-lo.

### **Idioma**

Os commits deste projeto devem ser escritos **preferencialmente em português**.

Isso faz parte da identidade do projeto e mantém seu histórico consistente com seu propósito de construir uma comunidade brasileira de aprendizado.

Exemplos:

```text
novo(regressao): adicionando implementação de regressão linear
docs(classificacao): melhorando explicação de matriz de confusão
corr(pca): corrigindo cálculo da variância explicada
refat(colabs): reorganizando notebooks de regressão
```

Ferramentas de Inteligência Artificial podem auxiliar na formulação dos commits, mas a responsabilidade pela escolha do tipo, escopo e descrição permanece com o contribuidor.

---

## 7. **Estrutura dos arquivos**

Cada tópico deve seguir, o máximo possível, a estrutura:

```text
topico/
├── README.md
└── colabs/
    ├── MAPA.md
    └── *.ipynb

docs/_topico
└── GUIA.md
```

### `README.md`

Apresenta um resumo rápido do tópico.

Deve conter:

* breve descrição do assunto;
* principais conceitos abordados;
* referências e materiais utilizados;
* links relevantes para outras partes do projeto.

O README deve ser objetivo. Ele funciona como uma porta de entrada para o tópico.

### `GUIA.md`

É a principal página de estudo do tópico.

O arquivo deve apresentar o conteúdo de maneira progressiva, conduzindo o leitor desde a introdução até os conceitos mais avançados abordados naquele tópico.

**Pode conter:**

* explicações;
* fórmulas;
* gráficos;
* imagens;
* exemplos;
* trechos de código;
* links para aulas;
* referências bibliográficas;
* links para notebooks.

A estrutura de **introdução → desenvolvimento → conclusão** deve estar presente de maneira natural, garantindo uma evolução incremental do conhecimento.

### `colabs/MAPA.md`

Reúne os notebooks disponíveis naquele tópico.

Cada entrada deve apresentar, de forma simples:

* título;
* descrição;
* link para o notebook.

Exemplo:

```md
## Predição de consumo em bares

A partir de uma base sintética de consumo de bebidas em um determinado bar, este notebook busca estimar, dado um conjunto básico de parâmetros, qual será o consumo de uma bebida x em determinado dia y. Isso permite estimar os custos previamente, otimizando gastos e evitando desperdício.

Acesse em: link_para_colab
Arquivo no projeto: caminho_arquivo

```

### `*.ipynb`

Os notebooks são o espaço de experimentação prática.

Devem conter explicações, mesmo que breves, sobre as implementações realizadas.

Sempre que possível:

* explique o objetivo do experimento;
* documente as principais etapas;
* justifique decisões importantes;
* interprete os resultados;
* inclua bibliografia básica;
* diferencie claramente código de biblioteca e implementações próprias.

A estrutura específica do notebook pode variar conforme o objetivo do experimento.

---

## 8. **Pull Requests**

Antes de abrir um Pull Request:

1. Verifique se a contribuição está de acordo com a proposta do projeto.
2. Revise o conteúdo e o código.
3. Execute os notebooks quando possível.
4. Utilize commits organizados.

Um Pull Request deve explicar brevemente **o que foi alterado e por quê**.

Contribuições podem passar por revisão e discussão antes de serem incorporadas ao projeto.

---

## 9. **Dúvidas e sugestões**

Encontrou um problema, possui uma sugestão ou gostaria de discutir uma ideia?

Abra uma **Issue** ou participe das discussões do projeto.

Toda contribuição pode ajudar a tornar este material melhor para a próxima pessoa que chegar aqui.
