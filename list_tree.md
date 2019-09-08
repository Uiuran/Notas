Uses -I for ignore specific pattern
```bash
tree -I '*.pyc'
tree -I '.*.swp'
```

```bash
ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'
``` 
