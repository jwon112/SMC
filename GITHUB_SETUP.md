# GitHub 업로드 가이드

SMC 프로젝트를 GitHub에 업로드하는 방법입니다.

## 방법 1: GitHub에서 리포지토리 먼저 생성 (권장)

### 1단계: GitHub에서 리포지토리 생성

1. GitHub 웹사이트 (https://github.com) 접속
2. 우측 상단 "+" 버튼 클릭 → "New repository"
3. 리포지토리 정보 입력:
   - Repository name: `SMC-classification` (또는 원하는 이름)
   - Description: "심장 병리 이미지 분류 실험 프로젝트"
   - Public/Private 선택
   - **중요**: "Add a README file", "Add .gitignore", "Choose a license" 모두 체크 해제
4. "Create repository" 클릭

### 2단계: 로컬에서 Git 초기화 및 업로드

```bash
# SMC 폴더로 이동
cd SMC

# Git 초기화
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: SMC classification project with multi-label support"

# 메인 브랜치로 이름 변경
git branch -M main

# GitHub 리포지토리 연결 (your-username을 실제 사용자명으로 변경)
git remote add origin https://github.com/your-username/SMC-classification.git

# 업로드
git push -u origin main
```

## 방법 2: 로컬에서 먼저 생성 후 GitHub에 연결

### 1단계: 로컬에서 Git 초기화

```bash
# SMC 폴더로 이동
cd SMC

# Git 초기화
git init

# 모든 파일 추가
git add .

# 첫 커밋
git commit -m "Initial commit: SMC classification project with multi-label support"

# 메인 브랜치로 이름 변경
git branch -M main
```

### 2단계: GitHub에서 리포지토리 생성

1. GitHub 웹사이트에서 새 리포지토리 생성 (방법 1의 1단계 참조)
2. **중요**: "Add a README file" 등은 체크하지 않음 (로컬에 이미 있음)

### 3단계: Remote 추가 및 Push

```bash
# GitHub 리포지토리 연결
git remote add origin https://github.com/your-username/SMC-classification.git

# 업로드
git push -u origin main
```

## 주의사항

1. **데이터 파일**: `.gitignore`에 의해 이미지 파일(`*.jpg`, `*.json`)은 제외됩니다.
   - CSV 파일만 포함됩니다 (라벨 정보)
   - 실제 이미지 데이터는 별도로 관리해야 합니다

2. **결과 파일**: `results/` 폴더와 `*.pth` 파일도 제외됩니다.

3. **인증**: GitHub에 push할 때 인증이 필요할 수 있습니다:
   - Personal Access Token 사용
   - 또는 SSH 키 설정

## 업데이트 방법

코드를 수정한 후:

```bash
cd SMC
git add .
git commit -m "Update: 설명"
git push
```

