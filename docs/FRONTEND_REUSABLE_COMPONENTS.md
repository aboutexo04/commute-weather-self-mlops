# 재사용 가능한 프론트엔드 컴포넌트 가이드

이 문서는 현재 출퇴근 날씨 웹앱의 프론트엔드 컴포넌트들을 다른 프로젝트에 재사용하기 위한 가이드입니다.

## 🎨 UI 컴포넌트 구조

### 1. 메인 버튼 레이아웃
```html
<!-- 시간대별 예측 버튼 그룹 -->
<div class="button-container">
    <button class="prediction-btn morning" onclick="getMorningPrediction()">
        🌅 출근길 예측
    </button>
    <button class="prediction-btn evening" onclick="getEveningPrediction()">
        🌆 퇴근길 예측
    </button>
    <button class="prediction-btn current" onclick="getCurrentWeather()">
        🌤️ 지금 날씨
    </button>
</div>
```

**CSS 스타일:**
```css
.button-container {
    display: flex;
    flex-direction: column;
    gap: 15px;
    margin: 20px 0;
}

.prediction-btn {
    padding: 15px 25px;
    font-size: 18px;
    border: none;
    border-radius: 10px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
}

.prediction-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
}
```

### 2. 결과 표시 카드
```html
<!-- 예측 결과 카드 -->
<div class="result-card" id="result">
    <h2 class="result-title">🌅 출근길 예측</h2>
    <div class="score-section">
        <div class="main-score">
            <span class="score-icon">⚠️</span>
            <span class="score-value">45.2</span>
            <span class="score-max">/100</span>
            <span class="score-label">(불편)</span>
        </div>
    </div>

    <div class="breakdown-section">
        <h3>📊 세부 점수:</h3>
        <div class="breakdown-item">
            <span class="category">온도:</span>
            <span class="penalty">-15.0점</span>
        </div>
        <div class="breakdown-item">
            <span class="category">강수:</span>
            <span class="penalty">-25.0점</span>
        </div>
        <div class="breakdown-item">
            <span class="category">바람:</span>
            <span class="penalty">-10.0점</span>
        </div>
        <div class="breakdown-item">
            <span class="category">습도:</span>
            <span class="penalty">-4.8점</span>
        </div>
    </div>

    <div class="message-section">
        <p class="warning-message">불편한 출근날씨입니다. 대비하세요! 😣</p>
    </div>
</div>
```

**CSS 스타일:**
```css
.result-card {
    background: white;
    border-radius: 15px;
    padding: 25px;
    margin: 20px 0;
    box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    border-left: 5px solid #4CAF50;
}

.score-section {
    text-align: center;
    margin: 20px 0;
}

.main-score {
    font-size: 28px;
    font-weight: bold;
}

.score-icon {
    font-size: 24px;
    margin-right: 10px;
}

.breakdown-section {
    margin: 20px 0;
}

.breakdown-item {
    display: flex;
    justify-content: space-between;
    margin: 8px 0;
    padding: 8px 0;
    border-bottom: 1px solid #eee;
}

.warning-message {
    text-align: center;
    font-size: 16px;
    font-weight: bold;
    margin: 15px 0;
    padding: 10px;
    border-radius: 8px;
    background-color: #fff3cd;
    border: 1px solid #ffeaa7;
}
```

## 🎯 점수 시스템 및 색상 코딩

### 점수 구간별 표시
```javascript
function getScoreDisplay(score) {
    let icon, label, colorClass;

    if (score >= 80) {
        icon = "🌟";
        label = "excellent";
        colorClass = "score-excellent";
    } else if (score >= 60) {
        icon = "🌟";
        label = "good";
        colorClass = "score-good";
    } else if (score >= 50) {
        icon = "⚠️";
        label = "uncomfortable";
        colorClass = "score-uncomfortable";
    } else {
        icon = "⚠️";
        label = "harsh";
        colorClass = "score-harsh";
    }

    return { icon, label, colorClass };
}
```

### 색상 테마
```css
.score-excellent { color: #27ae60; }
.score-good { color: #3498db; }
.score-uncomfortable { color: #f39c12; }
.score-harsh { color: #e74c3c; }

/* 카드 테두리 색상 */
.result-card.excellent { border-left-color: #27ae60; }
.result-card.good { border-left-color: #3498db; }
.result-card.uncomfortable { border-left-color: #f39c12; }
.result-card.harsh { border-left-color: #e74c3c; }
```

## 📱 반응형 디자인

### 모바일 최적화
```css
@media (max-width: 768px) {
    .button-container {
        padding: 0 10px;
    }

    .prediction-btn {
        font-size: 16px;
        padding: 12px 20px;
    }

    .result-card {
        margin: 10px;
        padding: 20px;
    }

    .main-score {
        font-size: 24px;
    }
}

@media (max-width: 480px) {
    .breakdown-item {
        font-size: 14px;
    }

    .result-title {
        font-size: 18px;
    }
}
```

## 🔧 JavaScript 함수 구조

### API 호출 함수 템플릿
```javascript
async function getPrediction(endpoint) {
    try {
        showLoading();
        const response = await fetch(`/api/${endpoint}`);
        const data = await response.json();
        displayResult(data);
    } catch (error) {
        displayError('예측을 가져오는데 실패했습니다.');
    } finally {
        hideLoading();
    }
}

function displayResult(data) {
    const resultDiv = document.getElementById('result');
    const { icon, label, colorClass } = getScoreDisplay(data.comfort_score.score);

    resultDiv.innerHTML = `
        <h2 class="result-title">${data.title}</h2>
        <div class="score-section">
            <div class="main-score ${colorClass}">
                <span class="score-icon">${icon}</span>
                <span class="score-value">${data.comfort_score.score.toFixed(1)}</span>
                <span class="score-max">/100</span>
                <span class="score-label">(${getKoreanLabel(label)})</span>
            </div>
        </div>
        ${generateBreakdown(data.comfort_score.penalties)}
        <div class="message-section">
            <p class="warning-message">${data.message}</p>
        </div>
    `;

    resultDiv.className = `result-card ${label}`;
    resultDiv.style.display = 'block';
}
```

### 로딩 상태 관리
```javascript
function showLoading() {
    const buttons = document.querySelectorAll('.prediction-btn');
    buttons.forEach(btn => {
        btn.disabled = true;
        btn.innerHTML = btn.innerHTML.replace(/🌅|🌆|🌤️/, '⏳');
    });
}

function hideLoading() {
    const buttons = document.querySelectorAll('.prediction-btn');
    buttons.forEach(btn => {
        btn.disabled = false;
    });
    // 원래 텍스트로 복원
    document.querySelector('.morning').innerHTML = '🌅 출근길 예측';
    document.querySelector('.evening').innerHTML = '🌆 퇴근길 예측';
    document.querySelector('.current').innerHTML = '🌤️ 지금 날씨';
}
```

## 🔄 ML 모델 적용을 위한 수정 포인트

### 1. API 엔드포인트 변경
```python
# 기존: 휴리스틱 모델
@app.get("/api/morning-prediction")
async def morning_prediction():
    predictor = CommutePredictor(kma_config)
    return predictor.predict_morning_commute()

# 변경: ML 모델
@app.get("/api/morning-prediction")
async def morning_prediction():
    # ML 모델 호출로 변경
    model_result = ml_model.predict(weather_features)
    return format_ml_result(model_result)
```

### 2. 데이터 구조 통일
```python
# ML 모델 결과를 기존 UI 구조에 맞게 변환
def format_ml_result(ml_prediction):
    return {
        "prediction_time": datetime.now(),
        "target_period": "morning_commute",
        "comfort_score": {
            "score": ml_prediction.comfort_score,
            "label": get_comfort_label(ml_prediction.comfort_score),
            "penalties": {
                "temperature": ml_prediction.temp_penalty,
                "precipitation": ml_prediction.precip_penalty,
                "wind": ml_prediction.wind_penalty,
                "humidity": ml_prediction.humidity_penalty
            }
        },
        "message": generate_message(ml_prediction.comfort_score)
    }
```

## 💡 재사용 시 체크리스트

- [ ] API 엔드포인트 URL 변경
- [ ] 브랜딩 요소 수정 (제목, 색상, 로고)
- [ ] 메시지 텍스트 프로젝트에 맞게 조정
- [ ] 점수 구간 및 라벨 검토
- [ ] 모바일 반응형 테스트
- [ ] 로딩 상태 및 에러 처리 확인
- [ ] 데이터 구조 호환성 검증

## 🚀 배포 고려사항

- **정적 파일**: CSS, JavaScript 파일 분리 고려
- **CDN**: 아이콘 폰트나 외부 리소스 확인
- **브라우저 호환성**: 최신 JavaScript 기능 사용 여부 점검
- **성능 최적화**: 이미지 압축, CSS/JS 압축 적용

이 컴포넌트들을 새 프로젝트에 적용하면 일관된 UI/UX를 유지하면서 ML 모델의 예측 결과를 효과적으로 표시할 수 있습니다.