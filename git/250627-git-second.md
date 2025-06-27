# 오늘 한 것

git merge conflict solve / github flow / troubleshoot / Co-work

# github flow

* git flow   
    * 응용프로그램, 어플리케이션 개발 시 많이 사용   
    * master branch와 develop branch 중요 -> 쉽게 건들지 말 것   
* github flow   
    * pull request   
    * branch에서 메인으로 pull request   
    * code rivew and merge   
    * git (main)에서 ls로 파일 확인 후 $ git pull origin main으로 리모트에서 가져오기   
* gitlab flow   
    * github flow가 단순한 것 같아서 만들었음   
    * 개발 단계와 검증 단계 나누기   

# troubleshoot   

## Stash : 작업 중 브랜치 이동 시 필요 / 작업 사항 잠시 미루기

* 작업 중인 변경사항 임시저장
```
$ git stash
```
* 임시저장 해둔 작업 복구
```
<!-- pop은 index 기준 뒷번호부터 추출되니 필요한 건 index 번호로 호출 -->   
<!-- stash한 거 pop하고 다시 stash할 수는 있으나, index 번호가 같은 건 보장 못함 -->
$ git stash pop   
$ git stash pop {index}
```
* 작업사항 리스트 확인
```
$ git stash list
```
* 작업 사항 삭제
```
$ git stash drop {index}
```   

## Undo : 변경사항 취소

* working directory에 있는 파일 변경사항 취소
```
$ git restore 파일명
```   

## Unstaging : stage의 변경사항 working directory로 내리기
* $ git add 파일명 으로 올라간 파일 내리기
```
$ git reset HEAD 파일명
```
* 내림과 동시에 삭제
```
$ git rm -f 파일명
```   

## Edit commit message
* 직전 commit message 수정
```
<--! --amend는 직전 커밋 재작성 ; 이전 커밋과 amend 후 커밋의 아이디가 다름 -->   
$ git commit --amend
```
* 이전에 했던 commit message 수정
```
$ git rebase -i <ommit>   
$ git rebase --continue   
<!-- rebase 취소 : $ git rebase --abort -->
```   

## 없애거나 되돌리기
* reset : 없던 일로 하기
```
<!-- 충돌 위험 -->
$ git reset --hard HEAD~없던일로만들커밋수
```
* revert : 되돌리기    
* revert 하면 잘못하기 전 시험으로 돌아가는데 그만큼 계속 커밋해야 함   
* 마지막에 커밋하겠단 의미 : --no-commit
```
$ git revert --no-commit HEAD~되돌릴커밋수   
<!-- 되돌리고 잘 고친 후 다시 commit -->   
$ git commit   
<!-- remote 브랜치에 push -->   
$ git push origin 리모트_브랜치명   

<!-- merge commit 되돌리기 -->   
$ git revert -m {1 or 2}머지커밋아이디
```   

# Co-work
**팀장**   
* 새 organization 생성   
* 팀원 초대 : member는 코드에 직접적 기여 불가, owner는 기여 가능   
    * 모든 사람이 기여하면 merge conflict 발생 ; 제한적 룰 두기   
* Repositories 새로 생성 후 clone과 .gitignore commit   
* 대상 파일 만들어서 add-commit-push     
* Issue templete 만들어두기   
    * Description / Tasks / References   
    * issue templete도 commit   
* milestone이나 project 설정해두면 좋음   
   
**팀원**   
* github 가입했던 메일로 온 초대링크 join   
* repo 클릭 후 issue 만들기   
    * 오른쪽 탭의 assignee, milestone 등은 팀장이 지정   
* clone 대신 fork 하기   
* 내 repo로 fork 한 것 clone   
* clone 한 것 새 브랜치 만들어, 새 브랜치에서 작업   
    * 작업 과정 별로 add-commit 진행    
    * 마지막에 push 하기   
* push 하기 : $ git push -u origin 내가만든브랜치   
* remote에서 pull request 하기   
* pull request : pr 제목 적고 내용에 close/fix/resolve #이슈번호 로 pr과 이슈 연결   
    * assignee는 pr한 사람, reviewer는 다른 사람   
    * review 다 하고 상단의 초록색 finish review 누르고 상태 고르기   
   
**마무리**   
* pr 승인 후 merge <팀장>   
* 팀 repo 주소 복사 후 $ git remote add upstream 팀주소 로 업데이트   
* git fetch upstream main    
* git merge FETCH_HEAD 로 merge하기   
     
**merge conflict**   
* remote에서 local로 오류 파일 가져오기   
    * $ git pull origin main   
* pull 하고 오류 파일 찾아 conflict 해결   
* add-commit-push 하면 pr에 자동으로 반영
