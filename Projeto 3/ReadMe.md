# Introdução
Para a realização deste projeto, em busca de um problema real, fez-se uso de uma competição existente no Kaggle, presente neste [link](https://www.kaggle.com/competitions/optiver-realized-volatility-prediction/) cujo objetivo é prever a volatilidade de um conjuntode ações num intervalo de 10 minutos. Para tal, foram fornecidos 3 diferentes dataframes com granularidades diferentes:

  * Um dataframe no qual estão as ordens de compra e de venda e em cada linha possui as duas ofertas maiores para venda e as duas menores ofertas de compra, com suas respectivas quantidades de ações solicitadas, além de informações relacionadas ao tempo de espera da ordem e o identificador da ação e o identificador do tempo.
  * Outro dataframe possui informações das operações que de fato foram executadas, com o tamanho, valor, tempo até execução e identificador da ação e o identificador do tempo.
  * E por fim, um dataframe com a volatilidade esperada , o identificador da ação e o identificador do tempo

# Metodologia
O primeiro passo para a realização do projeto foi o entendimento dos dataframes para uma única ação, diminuindo assim a quantidade de linhas. Dentro desse contexto,afim de otimizar os processos criou-se funções,uma de agragação das informações e outra de modelagem para facilitar a compreensão do programa. A função de agragação realiza Feature Engineering, com isso foram criadas novas variáveis de entrada para o modelo, e reduzidas a granularidade desejada, retornando um dataframe com as colunas de identificação e 43 novas colunas.

 Uma das etapas da função de modelagem é verificar a relevância das variáveis de entrada na variável de saída. Para isso, utilizou-se o modelo preditivo de [Lasso CV](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoCV.html), que permite esta verificação e reduziu para 37. Selecionadas as variáveis de entrada, a fim de verificar o melhor conjunto de hiperparametros para o modelo preditivo [CatBoostRegressor](https://catboost.ai/en/docs/), utilizou-se a biblioteca [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html) para otimizar os resultados. Esta função retona um dataframe com o identificador da ação (Stock_id), o identificador do tempo (Time_id), a volatilidade desejada (Target) e a volatilidade obtida (Predict) 

Para visualização dos resultados, usou-se uma outra biblioteca, a Shap, que permite a visualizção do impacto de cada uma das variáveis de entrada no resultado da predição. Já para calcular o erro do modelo, utilizpu-se o RMSPE (Root Mean Squared Percent Error).
# Resultados

O erro obtido no processo foi de 0,238 ou 23,8%, para todos os pontos do dataframe resultante da função modelo. O maior erro para uma única ação foi de 0,551 ou 55,1% para a ação identificada como Stock_id=31, e o menor erro encontrado foi de 0,164 ou 16,4% para ação identificada como Stock_id=56.

# Conclusão e Próximos Passos

Dataframes com diferentes granularidades necessitam de mais atenção para melhor entendimento. Além disso uma escolha razoável das variáveis de entrada impacta no tempo de execução, uma má escolha atrelada a otimização de hiper parâmetros são custosos computacionalmente, e podem fazer com o tempo para análise aumente consideravelmente.
 
