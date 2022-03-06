do a Procfile, in the same directory of .git repo with
web: python handler.py
in first line

heroku login

git init .

heroku create -a name-app

git remote add herokuappname https://git.heroku.com/name-app.git

git add *

git commit -m 'first commit'

git push herokuappname master
