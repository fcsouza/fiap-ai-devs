"""
Módulo com implementação da classe Portfolio para representação e análise de portfólios.

Este módulo define a classe principal que representa um portfólio de investimentos
e implementa métodos para avaliação de performance, exposição de ativos, 
risco e outras métricas relevantes.

"""
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

from src.metrics.performance import (
    calculate_metrics, calculate_information_ratio, calculate_sortino_ratio,
    calculate_calmar_ratio, calculate_treynor_ratio
)
from src.metrics.risk import (
    calculate_volatility, calculate_var, calculate_cvar, calculate_drawdown,
    calculate_diversification_ratio
)


class Portfolio:
    """
    Classe que representa um portfólio de investimentos com métodos para análise e avaliação.
    
    A classe Portfolio encapsula os dados e operações relacionadas a um portfólio de investimentos,
    fornecendo métodos para avaliação de performance, exposição de ativos, risco e retorno.
    
    Attributes:
        weights (np.ndarray): Pesos dos ativos no portfólio.
        assets (List[str]): Lista de identificadores dos ativos.
        returns (pd.DataFrame): Retornos históricos dos ativos.
        cov_matrix (pd.DataFrame): Matriz de covariância dos retornos.
        risk_free_rate (float): Taxa livre de risco anualizada.
        market_returns (Optional[pd.Series]): Retornos do mercado de referência.
    """

    def __init__(
        self,
        weights: np.ndarray,
        assets: List[str],
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None,
        risk_free_rate: float = 0.0,
        market_returns: Optional[pd.Series] = None
    ):
        """
        Inicializa um objeto Portfolio com os parâmetros fornecidos.
        
        Parameters
        ----------
        weights : np.ndarray
            Pesos dos ativos no portfólio.
        assets : List[str]
            Lista de identificadores dos ativos.
        returns : pd.DataFrame
            Retornos históricos dos ativos.
        cov_matrix : pd.DataFrame, optional
            Matriz de covariância dos retornos. Se None, será calculada.
        risk_free_rate : float, optional
            Taxa livre de risco anualizada.
        market_returns : pd.Series, optional
            Retornos do mercado de referência.
        
        Raises
        ------
        ValueError
            Se os tamanhos de weights e assets não forem compatíveis
            ou se os assets não estiverem presentes em returns.
        """
        if len(weights) != len(assets):
            raise ValueError("O número de pesos deve ser igual ao número de ativos.")

        if not all(asset in returns.columns for asset in assets):
            raise ValueError("Todos os ativos devem estar presentes nos dados de retorno.")

        # Normaliza os pesos para somar 1
        self.weights = weights / np.sum(weights)
        self.assets = assets
        self.returns = returns[assets]

        if cov_matrix is None:
            self.cov_matrix = self.returns.cov()
        else:
            if not all(asset in cov_matrix.columns for asset in assets):
                raise ValueError("Todos os ativos devem estar presentes na matriz de covariância.")
            self.cov_matrix = cov_matrix.loc[assets, assets]

        self.risk_free_rate = risk_free_rate
        self.market_returns = market_returns

    def __repr__(self) -> str:
        """Retorna uma representação em string do objeto Portfolio."""
        performance = self.get_performance_metrics()
        return (
            f"Portfolio(assets={len(self.assets)}, "
            f"return={performance['return']:.2%}, "
            f"risk={performance['volatility']:.2%}, "
            f"sharpe={performance['sharpe']:.2f})"
        )

    def get_weights_dict(self) -> Dict[str, float]:
        """
        Retorna um dicionário com os pesos de cada ativo.
        
        Returns
        -------
        Dict[str, float]
            Dicionário com pares {ativo: peso}.
        """
        return dict(zip(self.assets, self.weights))

    def get_volatility(self) -> float:
        """
        Calcula a volatilidade anualizada do portfólio.
        
        A volatilidade mede o risco total do portfólio, considerando as correlações
        entre os ativos e suas volatilidades individuais.
        
        Fórmula: Volatilidade = raiz(pesos × matriz covariância × pesos × 252)
        
        Returns
        -------
        float
            Volatilidade anualizada do portfólio (expressa como decimal, ex: 0.15 = 15%).
        """
        vol = np.sqrt(np.dot(self.weights.T, np.dot(self.cov_matrix * 252, self.weights)))
        return vol

    def get_return(self) -> float:
        """
        Calcula o retorno anualizado esperado do portfólio.
        
        O retorno esperado é a média ponderada dos retornos históricos médios 
        de cada ativo, multiplicada por 252 para anualização.
        
        Fórmula: Retorno Anualizado = (soma dos pesos × retornos médios) × 252
        
        Returns
        -------
        float
            Retorno anualizado esperado do portfólio (expresso como decimal, ex: 0.10 = 10%).
        """
        return np.sum(self.returns.mean() * self.weights) * 252
        
    def get_sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calcula o índice de Sharpe do portfólio.
        
        O índice de Sharpe é a principal métrica de eficiência risco-retorno,
        medindo o retorno excedente por unidade de risco total.
        
        Fórmula: Sharpe = (Retorno do portfólio - Taxa livre de risco) / Volatilidade
        
        Interpretação:
        - <0.5: Desempenho ruim
        - 0.5-1.0: Desempenho aceitável
        - 1.0-2.0: Desempenho bom
        - >2.0: Desempenho excelente
        
        Parameters
        ----------
        risk_free_rate : float, optional
            Taxa livre de risco anualizada, by default 0.0
        
        Returns
        -------
        float
            Índice de Sharpe do portfólio.
        """
        ret = self.get_return()
        vol = self.get_volatility()
        if vol == 0:
            return 0
        return (ret - risk_free_rate) / vol

    def get_performance_metrics(self, benchmark_returns=None, risk_free_rate=0.0):
        """
        Calcula métricas de performance do portfólio.
        
        Produz um conjunto completo de métricas para avaliar o desempenho do portfólio
        sob diferentes perspectivas (risco total, risco de queda, risco sistemático).
        
        Métricas calculadas:
        - Retorno anualizado: Retorno esperado do portfólio em base anual
        - Sharpe: Retorno excedente por unidade de risco total
        - Sortino: Retorno excedente por unidade de risco de queda
        - Calmar: Retorno excedente por unidade de drawdown máximo
        - Information Ratio: Retorno ativo por unidade de risco ativo (vs benchmark)
        - Treynor: Retorno excedente por unidade de risco sistemático (beta)
        
        Parâmetros:
            benchmark_returns (pd.Series): Retornos do benchmark (opcional).
            risk_free_rate (float): Taxa livre de risco (padrão: 0.0).
            
        Retorna:
            dict: Dicionário com métricas de performance.
        """
        from src.metrics.performance import (
            calculate_sortino_ratio, calculate_information_ratio,
            calculate_treynor_ratio, calculate_calmar_ratio
        )
        
        # Métricas básicas
        ret = self.get_return()
        sharpe = self.get_sharpe_ratio(risk_free_rate)
        
        # Métricas adicionais que não dependem do benchmark
        sortino = calculate_sortino_ratio(self.weights, self.returns, risk_free_rate)
        calmar = calculate_calmar_ratio(self.weights, self.returns, risk_free_rate)
        
        # Construir dicionário de resultados
        metrics = {
            'return': ret,
            'sharpe': sharpe,
            'sortino': sortino,
            'calmar': calmar
        }
        
        # Métricas que dependem do benchmark
        if benchmark_returns is not None:
            # Métricas que dependem do benchmark
            try:
                information_ratio = calculate_information_ratio(self.weights, self.returns, benchmark_returns)
                treynor = calculate_treynor_ratio(self.weights, self.returns, benchmark_returns, risk_free_rate)
                
                metrics['information_ratio'] = information_ratio
                metrics['treynor'] = treynor
            except Exception as e:
                # Em caso de erro, registra e define valores padrão
                print(f"Aviso: Não foi possível calcular métricas de benchmark: {e}")
                metrics['information_ratio'] = 0
                metrics['treynor'] = 0
        
        return metrics

    def get_risk_metrics(self) -> Dict[str, float]:
        """
        Calcula métricas de risco do portfólio.
        
        Esta função calcula um conjunto abrangente de métricas de risco
        para avaliar diferentes aspectos do risco do portfólio:
        
        Métricas calculadas:
        - volatility: Dispersão total dos retornos (risco total)
        - var_95: Value at Risk com 95% de confiança (perda potencial) 
        - cvar_95: Conditional VaR/Expected Shortfall (média das piores perdas)
        - max_drawdown: Maior queda histórica de pico a vale
        - skewness: Assimetria da distribuição de retornos (valores negativos indicam risco de cauda)
        - kurtosis: "Peso" das caudas da distribuição (valores altos indicam maior risco de eventos extremos)
        - diversification_ratio: Grau de diversificação efetiva do portfólio
        
        Interpretação:
        - Menores valores de volatilidade, VaR, CVaR e drawdown são desejáveis
        - Assimetria positiva é preferível (mais ganhos extremos que perdas extremas)
        - Diversification ratio maior indica melhor diluição de risco
        
        Returns
        -------
        Dict[str, float]
            Dicionário com as métricas de risco calculadas.
        """
        # Criar série de retornos do portfólio
        portfolio_returns = (self.returns * self.weights).sum(axis=1)

        metrics = {
            'volatility': calculate_volatility(self.weights, self.cov_matrix),
            'var_95': calculate_var(self.weights, self.returns, confidence_level=0.95),
            'cvar_95': calculate_cvar(self.weights, self.returns, confidence_level=0.95),
            'max_drawdown': calculate_drawdown(self.weights, self.returns)['max_drawdown'],
            'diversification_ratio': calculate_diversification_ratio(self.weights, self.cov_matrix)
        }

        # Adicionar métricas de cauda
        returns_array = portfolio_returns.values
        metrics['skewness'] = pd.Series(returns_array).skew()
        metrics['kurtosis'] = pd.Series(returns_array).kurtosis()

        return metrics

    def get_portfolio_returns(self) -> pd.Series:
        """
        Calcula a série temporal de retornos do portfólio.
        
        Returns
        -------
        pd.Series
            Série de retornos diários do portfólio.
        """
        return (self.returns * self.weights).sum(axis=1)

    def get_portfolio_cumulative_returns(self) -> pd.Series:
        """
        Calcula a série temporal de retornos cumulativos do portfólio.
        
        Returns
        -------
        pd.Series
            Série de retornos cumulativos do portfólio.
        """
        return (1 + self.get_portfolio_returns()).cumprod() - 1

    def get_drawdown_series(self) -> pd.DataFrame:
        """
        Calcula a série de drawdowns do portfólio ao longo do tempo.
        
        Returns
        -------
        pd.DataFrame
            DataFrame com série de drawdown.
        """
        return calculate_drawdown(self.weights, self.returns)
