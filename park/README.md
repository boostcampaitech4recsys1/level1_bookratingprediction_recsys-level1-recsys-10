#### 박건영_T4075
### level1_book_rating_prediction

1. EDA
    - language columns -> isbn 규칙 적용해서 재구성
    - 유저 별 평균, 책 별 평균 -> 새로 생성했으나 모델 성능 체크에서는 제외함
        * epoch가 늘어날수록 overfitting 되는 경향
2. FFM
    - field 안의 feature 종류 수, field의 개수, field의 종류(category, number, rank 등)에 따라 성능 차이가 보임
    - epoch가 늘어날수록 rmse가 낮아지는 것 확인 -> 한계점이 분명하여, 성능향상을 위해선 feature 추가에 대한 사항을 고려해야할 것
3. DCN
    - cross network : explicit 정보 반영
    - Deep network : implicit 정보 반영
4. DCN + CNN_FM
    - 추가적인 feature 형성을 위해 결함
    - image vector 생성 후 CNN_FM으로 학습 / 나머지 context data를 DCN으로 학습
    - DCN 코드 내부에 CNN_FM 반영
    - input : context data, image vector 
    - predict 파트에서 문제가 생겨서 실험 중지
5. etc : NCF
    - EDA 후, NCF 성능 2.174대 확인
    - DCN으로 explicit, implicit 정보 반영하게 되면서 실험 중지
    
    