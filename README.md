# 📊 TelecomX – Previsão de Churn

## 🔎 Visão Geral
Este projeto busca prever o **churn de clientes** (cancelamento de serviços) na **TelecomX**.  
A análise inclui exploração de dados, pré-processamento e modelagem de Machine Learning para apoiar **estratégias de retenção**.

---

## 🎯 Objetivos
- Tratar e preparar os dados (limpeza, encoding, normalização)  
- Balancear classes desiguais com **SMOTE**  
- Identificar variáveis mais relacionadas ao churn  
- Treinar e comparar modelos (Regressão Logística e Random Forest)  
- Avaliar desempenho e extrair insights de negócio  

---

## 🛠️ Metodologia

### 🔹 Preparação dos Dados
- Separação entre variáveis categóricas e numéricas  
- Remoção de colunas irrelevantes (`customerID`)  
- **Encoding:** OneHotEncoder para variáveis categóricas  
- **Normalização:** StandardScaler para variáveis numéricas  
- **Balanceamento:** SMOTE para corrigir desbalanceamento (≈26% churn)  
- **Divisão dos dados:** 70% treino | 30% teste (estratificado)  

### 🔹 Principais Descobertas
- Clientes **recém-chegados** têm maior risco de evasão  
- **Contrato mensal** é o fator mais crítico de churn  
- Usuários de **fibra óptica** cancelam com mais frequência  
- Pagamento via **cheque eletrônico** está associado a evasão  
- **Baixo gasto acumulado** é fortemente ligado ao cancelamento  

### 🔹 Modelagem
| Modelo                | Tipo                | Destaque |
|-----------------------|---------------------|----------|
| **Regressão Logística** | Linear, baseline    | Alto **recall** – identifica melhor clientes em risco |
| **Random Forest**       | Árvores, não linear | Maior **precisão** – previsões mais assertivas |

---

## 📈 Resultados
- **Regressão Logística** → indicada quando a prioridade é **não perder clientes em risco**, mesmo com falsos positivos.  
- **Random Forest** → recomendada para cenários que exigem **ações precisas**, reduzindo falsos alertas.  

### 🔑 Variáveis mais relevantes
- Tempo de contrato baixo (tenure)  
- Contrato mensal  
- Internet via fibra óptica  
- Pagamento por cheque eletrônico  
- Baixo gasto acumulado  

---

## ✅ Recomendações
- Criar ações de retenção para **clientes novos e de contratos mensais**  
- Incentivar **migração para planos de longa duração**  
- Revisar a experiência de clientes que usam **fibra óptica**  
- Oferecer **benefícios diferenciados** para quem paga com cheque eletrônico  
- Monitorar de perto clientes de **baixo valor acumulado**  

---

## ⚙️ Tecnologias Utilizadas
- **Linguagem:** Python 3.10+  
- **Bibliotecas:** Pandas, NumPy, Scikit-learn, imbalanced-learn, Matplotlib, Seaborn  
- **Ambiente:** Jupyter Notebook / VSCode  

---

## 🚀 Como Executar
1. Clone o repositório:
   ```bash
   git clone <url-do-repositorio>
   cd TelecomX-Churn

