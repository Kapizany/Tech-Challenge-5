# Adaptive Offers — FIAP Tech Challenge 5

Repositório público: [github.com/Kapizany/Tech-Challenge-5](https://github.com/Kapizany/Tech-Challenge-5). A versão local desta revisão ainda precisa de commit e push para aparecer no remote.

Serviço de recomendação adaptativa que escolhe, antes do contato, o canal (`cellular` ou `telephone`) de uma campanha de depósito a prazo. A API recebe contexto pré-contato, estima a recompensa esperada por canal, aplica limites de exposição e registra ação, propensão e versão da política. A recomendação requer revisão humana e não dispara contato.

O produto é um só. Não há mensagem nem oferta alternativa: a decisão do sistema é o canal. A solução não escolhe produto, preço, crédito ou público a partir de renda, patrimônio, gênero ou raça.

## Problema e impacto

Uma instituição digital precisa decidir qual canal usar para cada cliente elegível. O serviço combina uma política adaptativa com limites de exposição e comparação contínua com uma regra fixa. A evidência disponível não autoriza promover a política atual: o quality gate bloqueia a promoção até que o modelo e a avaliação temporal atendam aos critérios.

Nesta base o ganho não está provado. No treino, celular converte 5,83% e telefone 3,95%, então o baseline fixo é sempre celular. No último período da campanha o celular continua na frente (32,2% contra 21,0% no telefone). O modelo de recompensa, porém, prevê cerca de 66% para os dois canais e dá uma vantagem residual de 0,4 ponto percentual ao telefone em todas as linhas de teste. O ROC-AUC desse modelo no teste temporal é 0,41. A simulação semissintética mostra um uplift pequeno do Thompson Sampling contra o baseline porque o simulador obedece a esse modelo, não porque o telefone tenha convertido mais no futuro observado.

O relatório marca `deployment_claim = false`. Três condições falham ao mesmo tempo: o ranking do modelo discorda da conversão observada no teste, o ROC-AUC temporal fica abaixo de 0,5 e o OPE corta 31% das propensões. O resultado fica publicado. Ele não autoriza trocar o canal em produção.

## Base

Arquivo `bank-additional-full.csv`, correspondente à base [Bank Marketing, de henriqueyamahata no Kaggle](https://www.kaggle.com/datasets/henriqueyamahata/bank-marketing). O downloader usa por padrão a distribuição oficial do [UCI Machine Learning Repository, dataset 222](https://archive.ics.uci.edu/dataset/222/bank+marketing), sem credenciais; Kaggle é alternativa opcional para obter a mesma base. A referência Kaggle identifica a distribuição escolhida; a fonte primária UCI é usada para aquisição reproduzível. O estudo de referência é Moro, Cortez e Rita, *A Data-Driven Approach to Predict the Success of Bank Telemarketing*, Decision Support Systems, 2014.

- Licença do dado: CC BY 4.0, conforme o UCI.
- 41.188 linhas, 21 colunas, alvo `y` (assinou o depósito: `yes`/`no`).
- Conversão global: 11,27%.
- SHA-256 do arquivo usado: `74adfc578bf77a7ff4bb1ba4a9f8709d9e3c6907342959c2c8416847e0afb4d8`.
- O raw não entra no Git. `make data` baixa a cópia oficial do UCI. `make data-kaggle` usa o mesmo arquivo via Kaggle.

### Implementações e estado

| Área | Componentes | Estado |
|---|---|---|
| Exploração e qualidade | `notebooks/01_eda_preparation.ipynb`, `make eda-summary` | Schema, qualidade, conversão observada, segmentos, variação temporal, macroeconomia e diagnóstico de leakage. Não transforma features. |
| Preparação de dados | `notebooks/02_preparation_no_leakage.ipynb`, `make prepare` | Remove duplicatas, deriva atributos de decisão, separa contexto/ação/recompensa e ajusta transformações somente no treino; verificações executáveis e manifesto. |
| Políticas e avaliação | Scripts e notebooks de modelagem, avaliação e Golden Set | Thompson Sampling, Epsilon-Greedy, baselines, métricas offline, OPE e cinco casos explicáveis. O uplift de simulação não é causal; o quality gate bloqueia a promoção atual. |
| Serviço de recomendação | FastAPI, `/recommend`, `/feedback`, `/health`, `/model-info`, Docker | Validação, propensão, versão, idempotência, revisão humana, persistência e correlação por request. |
| Ciclo de aprendizagem | MLflow, feedback idempotente, `make consolidate-policy`, `make approve-policy` | Runs de treino e avaliação verificáveis; consolidação semanal em candidato; posterior ativo somente após quality gate e aprovação. |
| Infraestrutura | Terraform GCP e GitHub Actions | Cloud Run API/Job, Scheduler, armazenamento, mensageria e alertas declarados; workflows de CI, deploy e aprovação configurados. Nenhum recurso foi aplicado à conta GCP. Vertex AI Pipelines/Registry seguem pendentes. |
| Apresentação | Roteiro e exemplos neste repositório | O vídeo de até cinco minutos ainda precisa ser gravado. |

A revisão documental está no workspace e ainda precisa de commit e push para aparecer no repositório remoto. A política não é autorizada para uso operacional: o ranking do reward model discorda da conversão observada no teste temporal, ROC-AUC fica abaixo de 0,5 e o OPE tem 31% de clipping.

Limitações: campanha portuguesa de 2008 a 2010, um único produto, sem identificador de cliente, canal já escolhido antes do registro e sem o resultado do canal que não foi usado. A transportabilidade para clientes brasileiros é baixa.

## Governança

A base é dado público de pesquisa, não cadastro de clientes deste projeto. A base legal usada aqui é a licença CC BY 4.0 e a finalidade acadêmica de experimentar uma política de canal. Não há decisão de crédito, cobrança ou recusa de produto.

Minimização: fora do modelo ficam `duration` (só existe depois da ligação), identificadores, renda, patrimônio, gênero e raça. `job`, `marital`, `education` e `age` permanecem porque estão na base pública e são proxies; existe o comando `python scripts/prepare_data.py --without-proxies` para repetir a preparação sem eles.

Retenção: o CSV bruto e os artefatos grandes ficam fora do Git. A política de retenção prevê lifecycle na camada de objetos e limita a retenção do feedback à janela operacional aprovada. A API não grava o payload do cliente. A aplicação dos recursos cloud depende do provisionamento GCP, ainda pendente.

Humano no loop: `POST /recommend` devolve `human_review_required: true`. A resposta não autoriza o contato.

## O que a EDA muda na modelagem

O notebook [`notebooks/01_eda_preparation.ipynb`](notebooks/01_eda_preparation.ipynb) e `make eda-summary` registram os fatos que as análises públicas desta base já tinham mostrado, mais o que o split temporal muda.

| Recorte | Contatos | Conversão |
|---|---:|---:|
| Global | 41.188 | 11,3% |
| Celular | 26.144 | 14,7% |
| Telefone | 15.044 | 5,2% |
| `poutcome=success` | 1.373 | 65,1% |
| Estudante | 875 | 31,4% |
| Aposentado | 1.720 | 25,2% |
| Maio | 13.769 | 6,4% |
| Março | 546 | 50,5% |
| Dezembro | 182 | 48,9% |
| Setembro | 570 | 44,9% |
| Outubro | 718 | 43,9% |

A conversão cai com a repetição: 13,0% no primeiro contato da campanha e 5,5% a partir do sexto. `default='yes'` aparece 3 vezes. Há 12 duplicatas exatas; a preparação mantém a primeira. `pdays=999` marca “nunca contatado” em 39.673 linhas (96,3%) e não entra como distância. `campaign` inclui o contato atual; o modelo usa `contacts_before = campaign - 1`.

O gap de canal é em parte calendário. No primeiro quinto do arquivo, ordenado de maio de 2008 a novembro de 2010, só existe telefone e a conversão é 3,1%. No último quinto, celular converte 32,2% (7.245 contatos) e telefone 21,0% (993). Na faixa mais baixa de Euribor o gap é 26,1% contra 18,9%. Na faixa mais alta, 5,6% contra 4,7%. `euribor3m` e `nr.employed` correlacionam 0,95. Há um trecho intermediário, ainda dentro do treino, em que o telefone converte 13,1% (670 contatos) e o celular 5,2%. Esse recorte não se repete no teste.

`duration` é o vazamento clássico. Num split aleatório, o ROC-AUC sai de 0,801 sem duração para 0,942 com duração, na linha dos notebooks públicos. No split temporal deste projeto, os mesmos modelos caem para 0,464 e 0,533. O split aleatório mistura o fim da campanha no treino. O temporal mostra que o regime futuro não se prevê com o passado, mesmo com a duração.

## Preparação dos dados

Os notebooks têm responsabilidades distintas. `notebooks/01_eda_preparation.ipynb` audita o schema e explora qualidade, alvo, canais e variação temporal; `notebooks/02_preparation_no_leakage.ipynb` aplica as transformações e valida o conjunto de modelagem. A EDA não repete a construção das matrizes.

Na cópia auditada há 41.188 linhas; 12 duplicatas exatas são removidas, mantendo a ordem, resultando em 41.176 registros. O split cronológico, sem embaralhamento, produz 24.705 linhas de treino, 8.235 de validação e 8.236 de teste. Como não há timestamp completo por linha, a ordem da fonte é uma proxy temporal e essa suposição é registrada.

O contexto substitui `campaign` por `contacts_before = campaign - 1` e `pdays` por `never_contacted`/`pdays_contacted`, para tratar a tentativa atual e a sentinela `999`. `duration` é pós-contato e excluída; `contact` é a ação observada; `y` vira a recompensa binária. `unknown` permanece categoria e `no` permanece resultado válido. O one-hot, a imputação numérica e a escala são ajustados apenas no treino. A preparação gera 57 features, matrizes `X_*`, vetores `actions_*`/`rewards_*`, `context_preprocessor.joblib` e `preparation_manifest.json` em `data/processed/`.

## Baseline, modelo e política

O baseline obrigatório é o canal com maior conversão no treino: celular, 5,83% (IC de Wilson 5,41% a 6,28%) contra telefone 3,95% (3,64% a 4,30%). Há também baseline aleatório e baseline por segmento (`poutcome`, faixa de contatos anteriores e `housing`). No teste, o segmento não troca o canal em relação ao baseline fixo.

O modelo de recompensa é uma regressão logística calibrada, com o canal como feature e o restante do contexto compartilhado. Dois modelos separados, um por canal, extrapolavam o Euribor de forma diferente e inventavam uma vantagem grande para o telefone. O modelo único evita isso, mas continua mal calibrado fora do treino: no teste, ROC-AUC 0,41, PR-AUC 0,28 e Brier 0,34, com probabilidade média prevista de 66,2% no celular e 66,6% no telefone.

Thompson Sampling usa essa probabilidade como média de uma Beta, com força 40 e prior Beta(1, 1). O contexto entra na média. Os guardrails de 5% a 95% limitam a exposição acumulada; a propensão registrada é calculada antes de atualizar essa exposição. Epsilon-Greedy usa a mesma probabilidade e `epsilon=0.10`. Priors, epsilon e a força do modelo estão em `artifacts/phase3_4_report.json` e nos experimentos MLflow `adaptive-offers-phase3-4` e `adaptive-offers-phase5-evaluation`.

### Simulação, 30 seeds, teste temporal

A recompensa contrafactual sai do modelo ajustado no treino. Não é a conversão observada.

| Política | Conversão simulada | Uplift absoluto vs. sempre celular | IC 95% do uplift |
|---|---:|---:|---:|
| Epsilon-Greedy contextual | 0,6654 | +0,0039 | 0,0037 a 0,0042 |
| Aleatória | 0,6635 | +0,0021 | 0,0019 a 0,0022 |
| Thompson Sampling contextual | 0,6635 | +0,0020 | 0,0018 a 0,0023 |
| Sempre o melhor canal do treino | 0,6615 | 0 | — |
| Segmento determinístico | 0,6615 | 0 | — |

O intervalo do Thompson fica acima de zero porque o simulador paga a vantagem residual de 0,4 ponto que o próprio modelo atribui ao telefone. A conversão observada no teste aponta o celular. Por isso o uplift não é efeito de canal.

### OPE no mesmo teste

IPS e Doubly Robust saem do intervalo de uma taxa de conversão. O clipping é 31% e o effective sample size do baseline fica em 2.713 de 8.236 linhas. SNIPS, o estimador menos instável desta rodada, também fica abaixo do baseline para o Thompson:

| Política | SNIPS | IPS | Doubly Robust |
|---|---:|---:|---:|
| Baseline fixo / segmento | 0,505 | 15,34 | -3,83 |
| Aleatória | 0,449 | 7,90 | -2,98 |
| Thompson Sampling | 0,452 | 6,91 | -2,48 |
| Epsilon-Greedy | 0,197 | 1,20 | -2,20 |

## Cinco casos

Gerados de `splits.test` por `make golden-set`. O contexto não inclui `duration`, o canal observado nem `y`. As duas probabilidades ficam próximas; a ação servida pode diferir do rótulo do caso quando a diferença é menor que a exploração.

| Caso | Perfil | Celular | Telefone | Ação | Por que |
|---|---|---:|---:|---|---|
| `unknown_category` | 44 anos, blue-collar, maio, sem resultado anterior, 2º contato | 0,671 | 0,675 | celular | exploração; a média do telefone é só um pouco maior |
| `likely_cellular` | 55 anos, aposentado, junho, fracasso anterior | 0,657 | 0,657 | telefone | empate; o guardrail apontou o mesmo canal sorteado |
| `likely_telephone` | 24 anos, admin., julho, sucesso anterior, 1º contato | 0,485 | 0,513 | celular | exploração em torno de uma vantagem pequena do telefone |
| `high_uncertainty` | 30 anos, admin., junho, sucesso anterior | 0,644 | 0,644 | celular | empate e exploração |
| `edge_campaign` | 72 anos, aposentado, agosto, 16 contatos na campanha | 0,672 | 0,674 | telefone | maior recompensa esperada, por margem mínima |

Nenhum caso justifica contato automático. A pergunta de revisão é se o canal é coerente com a campanha, não se aquele cliente teria convertido.

## Execução local

Python 3.11 a 3.13.

```bash
python3 -m venv .venv
source .venv/bin/activate
make install
make data          # UCI, sem credencial
make validate
make prepare
make eda-summary   # tabelas em artifacts/eda_summary.json
make train-policies
make evaluate      # 30 seeds; smoke: make evaluate SEEDS=2
make golden-set
make notebooks     # pipeline + executa os quatro notebooks sem sobrescrever os originais
make test
make lint
make verify-mlflow # confirma parâmetros e métricas de treino/avaliação no tracking
make quality-gate  # bloqueia promoção se os critérios offline não forem satisfeitos
make recommend     # um cliente de exemplo no stdout
make api           # http://127.0.0.1:8000
make docker-build
make infra-validate
```

`make notebooks` roda o pipeline completo (download UCI, validação, preparação, treino, avaliação oficial de 30 seeds e Golden Set) e executa os quatro notebooks. As cópias com resultados ficam em `/tmp/tech-challenge-notebooks`, sem sobrescrever os arquivos originais. Para executar só a EDA, `make eda` executa o notebook `01` no lugar. Também é possível mudar o destino com `make notebooks NOTEBOOK_OUTPUT_DIR=./notebooks-executados`. O contrato da API está na seção seguinte. Suba o serviço com `make api` em `http://127.0.0.1:8000`.

## API

A API escolhe o canal de contato (`cellular` ou `telephone`) para um cliente elegível. Ela não escolhe produto e não dispara o contato: toda recomendação volta com `human_review_required: true`. O corpo não aceita `duration`, o canal já observado nem `y`.

Decisões e feedback ficam em SQLite (`artifacts/adaptive_offers.sqlite3`). `ADAPTIVE_STORE=firestore` usa Firestore entre réplicas. `FEEDBACK_TOPIC` e `GOOGLE_CLOUD_PROJECT` publicam o feedback no Pub/Sub depois de persistir. `FEEDBACK_BIGQUERY_TABLE` liga o endpoint interno que grava no BigQuery. `MODEL_URI` aponta para o `joblib` local ou `gs://`.

Toda resposta inclui `X-Request-ID` e `X-Process-Time-Ms`. O cabeçalho `X-Request-ID` é opcional; se vier vazio ou fora do padrão `[A-Za-z0-9._:-]{1,128}`, a API gera um UUID.

### `GET /health`

Informa se o modelo carregou e se o armazenamento responde. O Cloud Run usa esta rota como sonda. O HTTP continua `200` quando o modelo falta; o campo `status` distingue o caso.

Sem parâmetros.

```bash
curl -s http://127.0.0.1:8000/health
```

```json
{"status": "ok", "model_loaded": true, "decision_store_ready": true}
```

`status` pode ser `ok`, `model_missing` ou `store_unavailable`.

### `GET /`

Repete o corpo de `/health` quando o serviço está pronto. Se `status` não for `ok`, responde `503` com `{"detail": "model_missing"}` ou `{"detail": "store_unavailable"}`.

```bash
curl -s http://127.0.0.1:8000/
```

### `GET /model-info`

Lista os canais, a versão da política e a exigência de revisão humana. Sem modelo, responde `503` com `{"detail": "Modelo não carregado. Execute make train-policies."}`.

```bash
curl -s http://127.0.0.1:8000/model-info
```

```json
{
  "actions": ["cellular", "telephone"],
  "policy": "thompson-sampling-contextual-v1",
  "context_enters_decision": true,
  "human_review_required": true
}
```

### `POST /recommend`

Devolve um canal para o cliente. A mesma `Idempotency-Key` com o mesmo corpo devolve a decisão gravada. A mesma chave com outro corpo responde `409`. Sem a chave, a API gera um UUID e usa esse valor como `request_id`.

O guardrail de exposição olha as decisões já gravadas. Com um canal em 100% e o outro em 0%, a chamada seguinte força o canal ausente, `action_probability` fica `1` e `reason` fica `guardrail`. Com os dois canais já usados, a escolha volta ao Thompson Sampling até um canal passar de 95%.

Cabeçalhos:

| Campo | Obrigatório | Regra |
|---|---|---|
| `Content-Type: application/json` | sim | |
| `Idempotency-Key` | não | 1 a 128 caracteres em `[A-Za-z0-9._:-]`. Fora do padrão: `422` |
| `X-Request-ID` | não | Correlação. Não substitui a chave de idempotência |

Corpo. Todos os campos são obrigatórios. Nomes com ponto são os da base; a forma com underscore (`emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `nr_employed`) também é aceita.

| Campo | Tipo | Regra |
|---|---|---|
| `age` | inteiro | 0 a 120 |
| `job` | texto | `admin.`, `blue-collar`, `entrepreneur`, `housemaid`, `management`, `retired`, `self-employed`, `services`, `student`, `technician`, `unemployed`, `unknown` |
| `marital` | texto | `divorced`, `married`, `single`, `unknown` |
| `education` | texto | `basic.4y`, `basic.6y`, `basic.9y`, `high.school`, `illiterate`, `professional.course`, `university.degree`, `unknown` |
| `default` | texto | `yes`, `no`, `unknown` |
| `housing` | texto | `yes`, `no`, `unknown` |
| `loan` | texto | `yes`, `no`, `unknown` |
| `month` | texto | `jan` … `dec` |
| `day_of_week` | texto | `mon`, `tue`, `wed`, `thu`, `fri` |
| `campaign` | inteiro | ≥ 1. Inclui o contato que está sendo decidido |
| `pdays` | inteiro | Dias desde o contato anterior. `999` significa nunca contatado |
| `previous` | inteiro | ≥ 0 |
| `poutcome` | texto | `failure`, `nonexistent`, `success` |
| `emp.var.rate` | número | |
| `cons.price.idx` | número | |
| `cons.conf.idx` | número | |
| `euribor3m` | número | |
| `nr.employed` | número | |

Categoria desconhecida ou tipo inválido: `422`. Sem modelo: `503`.

```bash
curl -s http://127.0.0.1:8000/recommend \
  -H 'content-type: application/json' \
  -H 'Idempotency-Key: demo-request-001' \
  -d '{"age":33,"job":"admin.","marital":"married","education":"university.degree","default":"no","housing":"yes","loan":"no","month":"may","day_of_week":"mon","campaign":1,"pdays":999,"previous":0,"poutcome":"nonexistent","emp.var.rate":1.1,"cons.price.idx":93.994,"cons.conf.idx":-36.4,"euribor3m":4.857,"nr.employed":5191.0}'
```

Resposta `200`, neste cliente e com exposição ainda vazia:

```json
{
  "request_id": "demo-request-001",
  "action": "telephone",
  "action_probability": 0.5638071865309167,
  "expected_reward_by_action": {
    "cellular": 0.06301094054193558,
    "telephone": 0.07292233894215457
  },
  "policy_version": "thompson-sampling-contextual-v1",
  "reason": "posterior_sample",
  "human_review_required": true,
  "note": "A recomendação escolhe o canal de contato, não um produto. Decisão sensível exige revisão humana antes de qualquer acionamento."
}
```

`reason` é `posterior_sample` ou `guardrail`. A mesma chave devolve este JSON enquanto o corpo não mudar. Outra chave pode sortear o outro canal.

### `POST /feedback`

Grava se o contato converteu. O `request_id` tem de ser o de uma recomendação já gravada. Reenviar a mesma recompensa responde `duplicate` e não cria outra linha. Recompensa diferente responde `409`. Esta rota não altera a próxima oferta. A consolidação do posterior fica no job semanal descrito mais abaixo.

| Campo | Obrigatório | Regra |
|---|---|---|
| `request_id` | sim | 1 a 128 caracteres em `[A-Za-z0-9._:-]`. Desconhecido: `404` |
| `reward` | sim | inteiro `0` ou `1` |
| `timestamp` | não | ISO-8601 com fuso, por exemplo `Z`. Sem fuso: `422`. Se omitido, a API usa o instante UTC atual |

```bash
curl -s http://127.0.0.1:8000/feedback \
  -H 'content-type: application/json' \
  -d '{"request_id":"demo-request-001","reward":1,"timestamp":"2026-09-27T12:00:00Z"}'
```

```json
{"request_id": "demo-request-001", "status": "accepted"}
```

O segundo envio idêntico devolve `"status": "duplicate"`. Se `FEEDBACK_TOPIC` estiver definido e a publicação falhar depois da gravação local, a resposta é `503` com `{"detail": "Feedback persistido localmente, publicação assíncrona indisponível; tente novamente."}`.

### `POST /events/feedback`

Recebe o push do Pub/Sub e grava a mesma recompensa no BigQuery. No Cloud Run, o IAM restringe quem chama. Sem `FEEDBACK_BIGQUERY_TABLE`, responde `503`.

O corpo é o envelope do Pub/Sub. Obrigatório: `message.data`, Base64 do JSON de feedback (`request_id`, `reward` e `timestamp` opcional). Os demais campos do envelope são ignorados.

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8000/events/feedback \
  -H 'content-type: application/json' \
  -d '{"message":{"data":"eyJyZXF1ZXN0X2lkIjoiZGVtby1yZXF1ZXN0LTAwMSIsInJld2FyZCI6MSwidGltZXN0YW1wIjoiMjAyNi0wOS0yN1QxMjowMDowMFoifQ=="}}'
```

Sucesso: `204` sem corpo. Envelope inválido ou JSON que não cumpre o contrato de `/feedback`: `400` com `{"detail": "Malformed Pub/Sub feedback event"}`.

## Consolidação semanal da política

O feedback acumulado é consolidado por um job único. `make consolidate-policy` soma sucessos e tentativas por segmento e grava uma versão candidata em `artifacts/policy_versions/<versão>/`. O job não publica `current.json` e não altera a política servida. Após quality gate aprovado e revisão humana, `make approve-policy` publica `current.json`; a API usa as contagens aprovadas na Beta junto com o reward model existente. O `joblib` não é retreinado nesse ciclo.

O gatilho inicial é semanal, em `configs/policy_update.json`, com `trigger` `interval`, `interval_days` `7` e `min_new_rewards` `50`. Semana sem recompensa nova suficiente não gera versão. `volume` consolida assim que o mínimo de recompensas novas aparece, sem esperar os sete dias. `drift` também ignora o calendário, mas só gera candidato se a taxa de conversão cair pelo menos `drift_reward_drop` (0,05) ou se a fatia de um canal mudar pelo menos `drift_action_share` (0,20) em relação à versão anterior. `promote` fica `false`: o job grava uma versão candidata e não faz deploy. A promoção segue o quality gate e uma aprovação humana.

As decisões novas guardam somente o segmento derivado de `poutcome`, faixa de contatos anteriores e `housing`; não salvam o payload completo do cliente. Decisões antigas sem segmento são ignoradas pelo job e contabilizadas no recibo. O modelo de recompensa em `joblib` não é retreinado nesse job. O estado candidato e seu manifesto ficam em `artifacts/policy_versions/`; execuções sem amostra suficiente ou antes do intervalo mínimo deixam recibo `skipped` em `artifacts/policy_versions/receipts/`.

O job lê `configs/policy_update.json`. `promote` permanece `false`. Com `POLICY_VERSIONS_URI=gs://.../policy_versions`, recupera o último candidato do bucket e envia o novo; sem essa variável, grava só em disco. O Cloud Run Job declarado usa Firestore e o bucket; o Cloud Scheduler está configurado para segundas-feiras, 09:00 em `America/Sao_Paulo`. O bootstrap mantém a agenda pausada enquanto usa a imagem placeholder; o deploy aprovado a ativa com a imagem validada. O script decide se a semana produz candidato. Alertas de Cloud Monitoring cobrem 5xx da API e falha do job. Nenhum recurso GCP foi aplicado ainda.

`make register-model` aplica o mesmo quality gate da promoção. Com o relatório atual ele bloqueia o registro. `VERTEX_REGISTER=1` só chama o Vertex AI Model Registry quando o gate passa, e exige `pip install -e '.[vertex]'`, `GOOGLE_CLOUD_PROJECT`, `MODEL_URI` e `TRAIN_IMAGE`.

O deploy aplica a infraestrutura e atualiza a imagem do job sem mudar a revisão servida pela API. Depois cria uma revisão da API sem tráfego, testa `/health` na URL marcada, desloca 10% do tráfego e observa a revisão por `CANARY_OBSERVATION_SECONDS` (variável do GitHub, padrão 300 s), com health checks a cada 30 s. Só promove para 100% se o Cloud Logging não registrar 5xx dessa revisão. Uma falha no fluxo retorna o tráfego à revisão anterior. Esse rollout está configurado, mas ainda não foi ensaiado em GCP.

```bash
make consolidate-policy
# Apenas com quality gate aprovado e candidato revisado:
make approve-policy VERSION=policy-20260928T090000000000Z-1234abcd APPROVED_BY=seu-usuario
# Para o store compartilhado, com credenciais cloud disponíveis:
ADAPTIVE_STORE=firestore make consolidate-policy
```

## Nuvem

O Terraform em `infra/terraform` declara Cloud Storage privado e versionado para dados/modelos, BigQuery, Artifact Registry, Firestore, Pub/Sub com dead-letter queue, Cloud Run privado e identidades separadas para runtime, deploy e Terraform. GitHub Actions usa Workload Identity Federation (OIDC), sem chave JSON. `infra/bootstrap-state.sh` cria o bucket remoto de state e habilita a identidade de serviço Pub/Sub; requer `gcloud` autenticado e permissões administrativas no projeto.

Nenhum recurso foi aplicado na sua conta GCP. Para habilitar a automação: configure o bootstrap/state, preencha `infra/terraform/terraform.tfvars` (incluindo `state_bucket_name`) a partir do exemplo, autentique o Terraform local com credenciais administrativas e faça um primeiro `terraform apply` para criar os recursos e as identidades WIF. O provider WIF agora aceita tokens somente da branch `main`; proteja essa branch e configure os GitHub Environments `infra-plan`, `dev`, `staging` e `prod` com revisores conforme o risco. Só então cadastre no GitHub os outputs WIF/service-account e as variáveis `GCP_PROJECT_ID`, `GCP_REGION`, `TF_STATE_BUCKET` e `DATA_BUCKET`. `terraform-plan.yml` é manual na branch protegida `main` (não autentica PRs); `deploy.yml` também é manual e executa o quality gate antes de autenticar/aplicar. Neste dataset o gate corretamente barra a promoção; não reduza seus limites apenas para fazer o deploy passar. A identidade de apply ainda tem papéis amplos de provisionamento e requer revisão IAM antes de uso real. Vertex AI Pipelines/Model Registry, retreino completo do reward model e Secret Manager não estão implementados. Scheduler, alertas e canário/rollback estão declarados/configurados, mas ainda não foram executados em GCP.

## MLOps e o que ainda é humano

`make train-policies` e `make evaluate` registram parâmetros, métricas e artefatos no MLflow; o Makefile usa `mlflow.db` por padrão e aceita `MLFLOW_TRACKING_URI` externo. Cada run leva as tags `git_sha`, `git_dirty` e `data_sha256`; o treino registra a conversão do baseline por canal e os hiperparâmetros das políticas, e a avaliação registra conversão, uplift, OPE e os critérios do gate (`gate_*`). `make verify-mlflow` exige esses campos nos dois runs, incluindo a comparação entre baseline e Thompson Sampling. `make mlflow-ui` abre a interface em `http://127.0.0.1:5000`. `make quality-gate` bloqueia promoção sem evidência suficiente. O CI executa Ruff, pytest, build Docker, Terraform fmt/validate e um smoke real de dados, modelos e MLflow com duas seeds; não executa notebooks nem faz scan de imagem. `terraform-plan.yml`, `deploy.yml` e `policy-approval.yml` são manuais e usam ambientes protegidos em `main`. O deploy executa 30 seeds, preserva os artefatos MLflow no GitHub Actions e só autentica no GCP após o quality gate. Os testes unitários não baixam a base.

O vídeo de até cinco minutos ainda precisa ser gravado. Roteiro: problema e o recorte de canal (0:00–0:40); base, `duration`, `pdays` e o confundimento com o Euribor (0:40–1:30); baseline celular, Thompson contextual e a tabela em que o uplift simulado não vira claim (1:30–3:00); `make recommend` ou `POST /recommend` com um dos cinco casos (3:00–4:00); MLflow, os dois parágrafos de GCP e a revisão humana (4:00–4:40); limitação principal e o que não foi provado (4:40–5:00).

## Limitações

A base não observa o contrafactual. O primeiro período só tem telefone, então a propensão histórica do canal é quase uma função do tempo. O modelo compartilhado reduz a extrapolação separada por canal e ainda erra o nível de conversão no teste. IPS e Doubly Robust não são interpretáveis com 31% de clipping. Não há identificador para separar contatos da mesma pessoa. A política não deve sair do modo de revisão.
