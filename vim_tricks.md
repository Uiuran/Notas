# General Tips

Lock ctrl+s, unlock terminal flow ctrl+q

save .swap with new name and do vim -d fileO savedswap

Always do esc > :w  then esc > :q to save and exit

## Navigate between different VIM windows 

Your plugins will make use of window division like tmux does. The default comand to navigate between them is Ctrl+w.

# Plugin Installation

There are plenty of plugins in:

https://github.com/vim-scripts

Soon enough i will be posting my .vimrc configuration

I strongly recommend you do git clone in the plugin to install, this allows you to developed the pluging as you need it.

The instructions below are present in VundeVim/README.md, 

git clone https://github.com/VundleVim/Vundle.vim.git ~/.vim/bundle/Vundle.vim

You must include the following on Top of your ~/.vimrc.



```vim
set nocompatible              " be iMproved, required
filetype off                  " required

" set the runtime path to include Vundle and initialize
set rtp+=~/.vim/bundle/Vundle.vim

call vundle#begin()
" alternatively, pass a path where Vundle should install plugins
"call vundle#begin('~/some/path/here')

" let Vundle manage Vundle, required
Plugin 'VundleVim/Vundle.vim'

" The following are examples of different formats supported.
" Keep Plugin commands between vundle#begin/end.
" plugin on GitHub repo
Plugin 'tpope/vim-fugitive'
" plugin from http://vim-scripts.org/vim/scripts.html
" Plugin 'L9'
" Git plugin not hosted on GitHub
Plugin 'git://git.wincent.com/command-t.git'
" git repos on your local machine (i.e. when working on your own plugin)
Plugin 'file:///home/gmarik/path/to/plugin'

" All of your Plugins must be added before the following line
call vundle#end()            " required
``` 
Install Plugins:

Launch vim and run :PluginInstall

To install from command line: vim +PluginInstall +qall

(optional) For those using the fish shell: add set shell=/bin/bash to your .vimrc

# Navigate through your code functions, classes with taglist.vim

You will need to install exuberant-ctags in your system:
```bash
sudo apt-get install exuberant-ctags
```

make sure the path you installed exuberant-ctags(just see the path to the binary in the verbose given in the command above, usually /usr/bin/) is in PATH environment variable ```bash printenv```

then put the Plugin as shown above, between the call vundle functions
```vim
Plugin 'git://github.com/vim-scripts/taglist.vim.git'
```

Also, you can remap the activation of your tag windows like this:
```vim
nnoremap <silent><C-b> :TlistToggle<CR>
``` 

Where ```vim <C-b>``` is Ctrl+b
