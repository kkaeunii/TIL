# 조건문    
## if 조건문     
| 조건문 형식 |           
| :--- |              
| if 조건문 : <br> &nbsp;&nbsp;&nbsp;조건이 참일 때 수행할 문장<br> else : <br> &nbsp;&nbsp;&nbsp;조건이 거짓일 때 수행할 문장 |       
| if 조건문 : <br> &nbsp;&nbsp;&nbsp;조건이 참일 때 수행할 문장 1 <br> elif 조건문 : <br> &nbsp;&nbsp;&nbsp;조건이 참일 때 수행할 문장 2 <br> elif 조건문 : <br> &nbsp;&nbsp;&nbsp;조건이 참일 때 수행할 문장 3 <br> else : <br> &nbsp;&nbsp;&nbsp;조건이 거짓일 때 수행할 문장               
* in과 not in을 넣고 조건문 만들 수 있음        
               
* 관계 연산자       
    * ==       
    * !=          
    * <         
    * \>               
    * <=                
    * \>=                   
                       
* 논리 연산자          
    * not 부정 ; 반대로 출력          
    * and 두 개의 값이 모두 True일 때만 True                 
    * or 두 개의 값이 모두 False일 때만 False              
                               
# 반복문     
## for 반복문       
* 리스트, 튜플, 문자열         
                    
| 기본 구조 | 문자열 반복 | 리스트 반복 | 딕셔너리 반복 | 범위 반복 |         
| :--- | :--- | :---| :--- | :--- |        
| for i in range(n) <br> &nbsp;&nbsp;&nbsp;print('출력하고 싶은 말') | 변수 = '문자열' <br><br> for a in 변수: <br> &nbsp;&nbsp;&nbsp;print(a) | 변수 = ['리','스','트'] <br><br> for a in 변수: <br> &nbsp;&nbsp;&nbsp;print(a) | 변수 = {key1:value1, key2: value2} <br><br> for key in 변수: <br> &nbsp;&nbsp;&nbsp;print('반복 수행 문장') | for i in range(n): <br> &nbsp;&nbsp;&nbsp;print(i) |           
| n번 출력하고 싶은 말 반복 | 문자열 하나하나를 꺼내기 | 리스트 요소 하나씩 꺼내기 | 딕셔너리 value값 접근 | 기본이 0부터 n-1까지 정수 출력 <br> range(숫자, 두개)는 처음 숫자부터 마지막 숫자-1까지 정수 출력 <br> range(숫자, 세, 개)는 처음 숫자부터 가운데 숫자-1까지 마지막숫자 간격 |          
               
## while 반복문    
* 조건이 참인 동안 수행할 문장이 반복    
                     
| 기본 구조 |         
| :--- |            
| 변수 = 0 <br><br> while 변수 < n: <br> &nbsp;&nbsp;&nbsp;print('수행할 문장')<br> &nbsp;&nbsp;&nbsp; 변수 = 변수 + 1 |   
| 변수 값이 증가하여 n을 도달하면 반복이 종료 |   
