// Formulário - Envio para API
document.getElementById('transactionForm').addEventListener('submit', async function(e) {
  e.preventDefault();
  
  const submitBtn = this.querySelector('button[type="submit"]');
  const originalText = submitBtn.textContent;
  submitBtn.textContent = 'Analisando...';
  submitBtn.disabled = true;

  try {
    const formData = new FormData(this);
    const transactionData = {
      month: parseInt(formData.get('month')),
      day: parseInt(formData.get('day')),
      day_of_week: parseInt(formData.get('day_of_week')),
      hour: parseInt(formData.get('hour')),
      category: formData.get('category'),
      gender: formData.get('gender'),
      state: formData.get('state'),
      part_of_day: formData.get('part_of_day'),
      amt: parseFloat(formData.get('amt'))
    };

    const response = await fetch('http://localhost:5000/predict', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(transactionData)
    });

    const result = await response.json();
    showResult(result);

  } catch (error) {
    console.error('Erro:', error);
    showResult({
      status: 'error',
      message: 'Erro ao conectar com o servidor'
    });
  } finally {
    submitBtn.textContent = originalText;
    submitBtn.disabled = false;
  }
});

// Mostrar resultados
function showResult(result) {
  const resultSection = document.getElementById('resultSection');
  const resultContent = document.getElementById('resultContent');
  const resultCard = document.getElementById('resultCard');

  if (result.status === 'success') {
    const riskClass = `risk-${result.risk_level ? result.risk_level.toLowerCase() : 'medium'}`;
    
    resultContent.innerHTML = `
      <div class="result-header ${riskClass}">
        <h4>${result.message || 'Análise Concluída'}</h4>
        <div class="risk-level">Nível de Risco: ${result.risk_level || 'MÉDIO'}</div>
      </div>
      <div class="result-details">
        <div class="probability">
          <span class="label">Probabilidade Base:</span>
          <span class="value">${result.probability_base !== undefined ? result.probability_base.toFixed(2) + '%' : 'N/A'}</span>
        </div>
        <div class="probability">
          <span class="label">Probabilidade Final:</span>
          <span class="value">${result.probability_final !== undefined ? result.probability_final.toFixed(2) + '%' : 'N/A'}</span>
        </div>
        <div class="fraud-status">
          <span class="label">Status da Transação:</span>
          <span class="value ${result.fraud ? 'fraud' : 'normal'}">
            ${result.fraud ? '🚨 FRAUDE DETECTADA' : '✅ TRANSAÇÃO NORMAL'}
          </span>
        </div>
        <div class="api-status">
          <span class="label">Status da API:</span>
          <span class="value success">${result.status || 'success'}</span>
        </div>
        <div class="raw-data">
          <details>
            <summary>📊 Dados Completos da Resposta</summary>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.8rem; overflow-x: auto;">
${JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    `;
    
    resultCard.className = `result-card ${riskClass}`;
  } else {
    resultContent.innerHTML = `
      <div class="result-header risk-high">
        <h4>❌ Erro na Análise</h4>
      </div>
      <div class="result-details">
        <p>${result.message || 'Erro desconhecido'}</p>
        <div class="raw-data">
          <details>
            <summary>📊 Dados Completos do Erro</summary>
            <pre style="background: #f8f9fa; padding: 10px; border-radius: 5px; margin-top: 10px; font-size: 0.8rem; overflow-x: auto;">
${JSON.stringify(result, null, 2)}
            </pre>
          </details>
        </div>
      </div>
    `;
    resultCard.className = 'result-card risk-high';
  }
  
  resultSection.style.display = 'block';
  resultSection.scrollIntoView({ behavior: 'smooth' });
}

// Botão limpar
document.getElementById('clearBtn').addEventListener('click', function() {
  document.getElementById('transactionForm').reset();
  document.getElementById('resultSection').style.display = 'none';
});

console.log('FraudGuard - Sistema carregado com sucesso!');