#!/usr/bin/env python
"""
JupyterLab + nbstripout のGit自動セットアップスクリプト

使用方法:
    python setup_git_jupyter.py

このスクリプトは以下を実行します:
1. nbstripout のインストール確認
2. Gitリポジトリの初期化
3. nbstripout の Git filter 登録
4. .gitattributes ファイルの生成
5. .gitignore ファイルの生成
"""

import os
import sys
import subprocess
import json
from pathlib import Path


def run_command(cmd, description):
    """コマンドを実行して結果を表示"""
    print(f"\n{'='*70}")
    print(f"📌 {description}")
    print(f"{'='*70}")
    print(f"実行: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.stdout:
            print(f"\n✅ 出力:\n{result.stdout}")
        
        if result.returncode != 0:
            print(f"⚠️  警告: {result.stderr}")
            return False
        
        return True
    
    except Exception as e:
        print(f"❌ エラー: {str(e)}")
        return False


def create_gitignore(project_dir):
    """
    .gitignore ファイルを生成
    """
    gitignore_path = Path(project_dir) / ".gitignore"
    
    gitignore_content = """# ============================================================
# Jupyter Notebook
# ============================================================
.ipynb_checkpoints/
.jupyter/
*.ipynb_checkpoints/
.jupyterlab/

# ============================================================
# Python
# ============================================================
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
env.bak/
venv.bak/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST

# ============================================================
# IDE & Editor
# ============================================================
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store
Thumbs.db

# ============================================================
# データファイル
# ============================================================
*.db
*.sqlite
*.csv
*.xlsx
*.parquet
*.pkl
*.pickle
*.h5

# ============================================================
# ローカル実験・テンポラリファイル
# ============================================================
/tmp/
/temp/
/experiments/
*.tmp
*.log
*.bak

# ============================================================
# OS固有ファイル
# ============================================================
.DS_Store
Thumbs.db
ehthumbs.db
Desktop.ini

# ============================================================
# その他
# ============================================================
.env
.env.local
secrets.json
config.local.json
"""
    
    try:
        with open(gitignore_path, 'w', encoding='utf-8') as f:
            f.write(gitignore_content)
        print(f"✅ .gitignore を生成しました: {gitignore_path}")
        return True
    except Exception as e:
        print(f"❌ .gitignore 生成エラー: {str(e)}")
        return False


def create_gitattributes(project_dir):
    """
    .gitattributes ファイルを生成
    """
    gitattributes_path = Path(project_dir) / ".gitattributes"
    
    gitattributes_content = """# Jupyter Notebooks
*.ipynb filter=nbstripout diff=ipynb merge=union
"""
    
    try:
        with open(gitattributes_path, 'w', encoding='utf-8') as f:
            f.write(gitattributes_content)
        print(f"✅ .gitattributes を生成しました: {gitattributes_path}")
        return True
    except Exception as e:
        print(f"❌ .gitattributes 生成エラー: {str(e)}")
        return False


def clean_notebook_metadata(notebook_path):
    """
    既存のnotebookメタデータをクリア
    """
    try:
        with open(notebook_path, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        
        # メタデータとセル出力をクリア
        for cell in nb.get('cells', []):
            cell['metadata'] = {}
            if 'execution_count' in cell:
                cell['execution_count'] = None
            if 'outputs' in cell:
                cell['outputs'] = []
        
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1)
        
        print(f"✅ Notebookのメタデータをクリアしました: {notebook_path}")
        return True
    
    except Exception as e:
        print(f"⚠️  Notebookメタデータクリアスキップ: {str(e)}")
        return False


def main():
    """メイン処理"""
    
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║  JupyterLab + Git + nbstripout 自動セットアップスクリプト  ║")
    print("╚" + "="*68 + "╝")
    
    # プロジェクトディレクトリの取得
    project_dir = input("\n📁 プロジェクトディレクトリのパスを入力してください\n   (デフォルト: 現在のディレクトリ): ").strip()
    
    if not project_dir:
        project_dir = os.getcwd()
    
    project_dir = os.path.abspath(project_dir)
    
    if not os.path.exists(project_dir):
        print(f"❌ エラー: ディレクトリが存在しません: {project_dir}")
        sys.exit(1)
    
    print(f"\n✅ 対象ディレクトリ: {project_dir}")
    
    # 作業ディレクトリを変更
    os.chdir(project_dir)
    
    # ============================================================
    # ステップ1: nbstripout のインストール確認
    # ============================================================
    print("\n\n【ステップ1】nbstripout のインストール確認")
    print("-" * 70)
    
    result = run_command(
        "pip list | find \"nbstripout\"" if os.name == 'nt' else "pip list | grep nbstripout",
        "nbstripout がインストール済みかチェック"
    )
    
    if not result:
        print("\n⚠️  nbstripout がインストールされていません。インストールします...\n")
        if not run_command("pip install nbstripout", "nbstripout をインストール"):
            print("❌ nbstripout のインストールに失敗しました")
            sys.exit(1)
    
    # ============================================================
    # ステップ2: Gitリポジトリの初期化
    # ============================================================
    print("\n\n【ステップ2】Gitリポジトリの初期化")
    print("-" * 70)
    
    git_dir = Path(project_dir) / ".git"
    if git_dir.exists():
        print(f"✅ Gitリポジトリが既に存在します: {git_dir}")
    else:
        if not run_command("git init", "Gitリポジトリを初期化"):
            print("❌ Gitリポジトリの初期化に失敗しました")
            sys.exit(1)
    
    # ============================================================
    # ステップ3: nbstripout の Git filter 登録
    # ============================================================
    print("\n\n【ステップ3】nbstripout の Git filter 登録")
    print("-" * 70)
    
    if not run_command("nbstripout --install", "nbstripout を Git filter として登録"):
        print("⚠️  警告: nbstripout のインストールに失敗しましたが、続行します")
    
    # ============================================================
    # ステップ4: .gitattributes ファイルの生成
    # ============================================================
    print("\n\n【ステップ4】.gitattributes ファイルの生成")
    print("-" * 70)
    
    create_gitattributes(project_dir)
    
    # ============================================================
    # ステップ5: .gitignore ファイルの生成
    # ============================================================
    print("\n\n【ステップ5】.gitignore ファイルの生成")
    print("-" * 70)
    
    gitignore_path = Path(project_dir) / ".gitignore"
    if gitignore_path.exists():
        print(f"⚠️  .gitignore は既に存在します: {gitignore_path}")
        overwrite = input("上書きしますか？ (y/n, デフォルト: n): ").strip().lower()
        if overwrite == 'y':
            create_gitignore(project_dir)
    else:
        create_gitignore(project_dir)
    
    # ============================================================
    # ステップ6: Notebook メタデータのクリア（オプション）
    # ============================================================
    print("\n\n【ステップ6】Notebookメタデータのクリア（オプション）")
    print("-" * 70)
    
    notebook_files = list(Path(project_dir).glob("*.ipynb"))
    if notebook_files:
        print(f"✅ {len(notebook_files)} 個の .ipynb ファイルが見つかりました:")
        for nb in notebook_files:
            print(f"   • {nb.name}")
        
        clean_option = input("\nメタデータをクリアしますか？ (y/n, デフォルト: n): ").strip().lower()
        if clean_option == 'y':
            for nb in notebook_files:
                clean_notebook_metadata(str(nb))
    else:
        print("ℹ️  Notebookファイル（.ipynb）が見つかりません")
    
    # ============================================================
    # ステップ7: 初回Gitコミット
    # ============================================================
    print("\n\n【ステップ7】初回Gitコミット（オプション）")
    print("-" * 70)
    
    commit_option = input("初回コミットを実行しますか？ (y/n, デフォルト: n): ").strip().lower()
    if commit_option == 'y':
        run_command("git add .", "ファイルをステージング")
        run_command(
            'git commit -m "Initial commit: set up Jupyter Git workflow with nbstripout"',
            "初回コミットを実行"
        )
    
    # ============================================================
    # 完了
    # ============================================================
    print("\n\n")
    print("╔" + "="*68 + "╗")
    print("║  ✅ セットアップが完了しました！             ║")
    print("╚" + "="*68 + "╝")
    
    print("\n【次のステップ】")
    print("1. JupyterLab を起動してください:")
    print("   jupyter lab")
    print("\n2. 左サイドバーの Git アイコン（赤いブランチ）をクリックしてください")
    print("\n3. ファイルを編集・保存してから、Git パネルでコミットしてください")
    print("\n【リモートリポジトリの設定（GitHub等）】")
    print("1. GitHub上で新しいリポジトリを作成してください")
    print("2. 以下のコマンドを実行してください:")
    print("   git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git")
    print("   git branch -M main")
    print("   git push -u origin main")


if __name__ == "__main__":
    main()
