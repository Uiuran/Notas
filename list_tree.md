
```bash
tree -I 'datasets|__pycache__|preprocessing|applications'
```

```bash
ls -R | grep ":$" | sed -e 's/:$//' -e 's/[^-][^\/]*\//--/g' -e 's/^/   /' -e 's/-/|/'
``` 
