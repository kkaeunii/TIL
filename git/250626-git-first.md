# git이란?

Linus Torvalds가 화나서 일주일만에 만든 형상관리 시스템(분산형)   
천재는 이럴 때 쓰는 말이었다.

# 오늘 한 것

경로 들어가기 cd 원하는 경로   
상위 경로 가기 cd ..

## git
```repo 만들고 clone 생성(git clone 깃허브url) -> .gitignore 설정하기!!!
```
- .gitignore : 특정 파일이나 디렉토리 추적하지 않게 함
```touch .gitignore > vi .gitignore로 조건 설정 적어두기   
```
[.gitignore](https://www.toptal.com/developers/gitignore/)

- vim이나 vi로 vim을 실행해도 됨

- 중요!! 움직임/수정/행동 단위마다 git add -> git commit -> (git push는 매번 해주지 않아도 된다고 함)
- git status로 확인해보기
- commit 할 때 제목 적어야 커밋됨
- push 처음 하면 입력창이 나오는데, 깃허브에서 token 발행해서 사용. 다음 push부터는 자동으로 됨.
- commit 메시지 설정 : 규칙대로 해두면 무엇을 했는지 알아보기 쉬움, 메시지는 수정 안 하는 게 나음.
```prefix를 꼭 달아야 함!!
```
    feat(기능 개발 관련) build(빌드 작업 관련) fix(오류 개선/패치) ci(Continous Intergration 관련) 
    docs(문서화) chore(패키지 매니저, 스크립트 등) test(test 관련) style(코드 포매팅 관련) 
    conf(환경설정 - .gitignore)

```config 설정
``` 
             $ git config --global user.name "{깃허브 사용자 이름}"
             $ git config --global user.email "{깃허브 가입한 메일}"
             $ git config --global core.editor "vim" ; vim으로 편집할 거니까
             $ git config --global core.pager "cat" ; cat으로 내용 볼 거니까
- config 값 보기 $ git config --list
- config 값 수정 $ git vi ~/.gitconfig -> 띄어쓰기 유의할 것. ~ /.으로 했다가 다른 화면 나왔었음;;

- pre-commit : commit 전 체크해야 할 것들 자동 수행-> python ssl오류로 실행하지 못함. python 삭제하고 다시 설치해보기

- branch!!! 분기점 생성하여 독립적으로 코드 변경 -> 무언가 잘못해도 branch 삭제하면 main엔 영향 없다
- git branch : branch 확인
- git switch 브랜치이름 : 주 브랜치 바꾸기
- git merge : 서로 다른 브랜치 병합 -> main에서 실행하기
    !! merge conflict 주의하기 -> conflict 일어난 것 전체 삭제 / 하나만 남겨두기 / 합의점 만들기
    git status로 conflict 일어난 파일 보기 > vi 파일명 conflict 해결

## Markdown
- # 이 표시 입력하는 걸로 이해했는데, 조금 더 공부가 필요할 것 같다.
- h1(#) ~ h6(######)
- 주석을 다는 것으로 생각하면 될까?

# 내일 할 것

git 협업
큰일남. 내 한몫 해야함.

# 소감

git 어려워요.. 잘 하고 싶어요..
