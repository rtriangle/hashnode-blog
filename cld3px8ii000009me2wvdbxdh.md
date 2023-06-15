---
title: "Multiple Python versions on Mac and how to set it up correctly"
datePublished: Thu Jan 19 2023 23:22:35 GMT+0000 (Coordinated Universal Time)
cuid: cld3px8ii000009me2wvdbxdh
slug: multiple-python-versions-on-mac-and-how-to-set-it-up-correctly
cover: https://cdn.hashnode.com/res/hashnode/image/stock/unsplash/toA-KC8Kwys/upload/746936b887bd3b23d849428adbddb96e.jpeg
tags: python, version-control, python3, python-beginner, python-projects

---

### Introduction

It is possible to have multiple versions of Python installed on a single MacBook. This can be done by using a version manager such as conda or using virtual environments with appropriate Python versions for each of your specific tasks. Sometimes it makes the usage of the specific version of Python, and alignment between installed packages and the pip version very confusing. You probably experienced the situation when you installed the Python package via pip and saw such an error during the validation of the successful installation:

```bash
python                                                                                                                             
Python 3.10.6 (main, Aug 30 2022, 05:12:36) [Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>> import <package_name>
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ModuleNotFoundError: No module named '<package_name>'
```

Let's dive deeper together into possible issues you may have and structure all the knowledge. There is nothing confusing at all!

### Which

First of all, to understand the number of possible Python versions installed on your Mac, you can simply check it in your bash console by running:

```bash
python <press Tab>

python python3-config python3.8 python3.9-config python.app python3.10 python3.8-config pythonw python3 python3.10-config python3.9
```

To see precisely the path to the Python interpreter that is used during the call (for my specific case): `python`, `python3.8`, etc.

```bash
which python
/usr/local/bin/python3.10

which python3.10            
/Users/daniilkorbut/anaconda3/bin/python3.8
```

Just a reminder, you can always type the full path to the Python interpreter and it will be the same as the shortcut name:

```bash
python3.10                                                                                                                                
Python 3.10.6 (main, Aug 30 2022, 05:12:36) [Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>

/usr/local/bin/python3.10                                                                                                                 
Python 3.10.6 (main, Aug 30 2022, 05:12:36) [Clang 13.1.6 (clang-1316.0.21.2.5)] on darwin
Type "help", "copyright", "credits" or "license" for more information.
>>>
```

### $PATH environment variable

And now we came to an important part, how to understand why `python3.10` was found in exact `/usr/local/bin/python3.10` directory. It means that the `/usr/local/bin` directory is included in the `$PATH` environment variable. We can view the content of this variable by running:

```bash
echo $PATH                                         
/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
```

`:` is a separator of the path in such syntax and the order of these paths defines the order in which the system tries to find the `python3` interpreter (for example) when you run `python3` without specification of `/usr/local/bin/python3` . You can always add the additional path to the `$PATH` variable to the beginning or the end, depending on your requirements. Depends on whether you use bash or zsh, you should have to edit `.bash_profile` or `.zshrc` for such purpose.

For zsh you should run `nano .zshrc` and add the necessary path to the new Python interpreter to the file. This example illustrates the addition of anaconda3 for both *append* and *prepend* options. For the append version, the new path for anaconda3 will have the lowest priority among existing ones, and for the prepend version - the highest priority.

```bash
# append
path+=('/Users/daniilkorbut/anaconda3/bin')
# or prepend
path=('/Users/daniilkorbut/anaconda3/bin' $path)
# export to sub-processes (make it inherited by child processes)
export PATH
```

For bash the same logic `nano .bash_profile` :

```bash
# append
PATH='${PATH}:/home/david/pear/bin'
# or prepend
PATH='/home/david/pear/bin:${PATH}'
export PATH
```

**Don't forget to restart the bash console after these changes!** Now, when you run `echo $PATH` the changes from the previous step should be reflected.

### Alias

One of the ways to redefine the Python interpreter path but **only for the active terminal session (the changes will go away after restart)** is the `alias` command. You can specify `alias python=python3.10` or `alias python=/usr/local/bin/python3.10` to use the *python* word instead of *python3.10* for your interpreter calls*.* To keep these changes for the next terminal session you can add the lines above to the already known `.zshrc` or `.bash_profile` files. If you don't want to use further aliased command, `unalias python` will revert the changes for the current session. In case of using aliases the output will be a bit different for `which` command:

```bash
alias py=python3.9

which py
py: aliased to python3.9

which python3.9
/usr/local/bin/python3.9
```

Another but very rare example of confusion that may appear, is the alignment between `pip` and `python` versions on the same machine. You can resolve it quickly by verifying specific **python** and **pip** locations on your machine:

```bash
which pip3.9                                      
/usr/local/bin/pip3.9

which python3.9                       
/usr/local/bin/python3.9
```

for this example the prefix of the path is the same which means `pip3.9 install <package name>` command will install the Python package for python3.9 interpreter located in `/usr/local/bin/python3.9` .

### Virtual environment

You can also use virtual environments to manage different versions of Python and their dependencies. Virtualenv is a tool that allows you to create isolated Python environments for different projects, which can help you avoid conflicts between packages that have different version requirements.

To install virtualenv, you can use pip:

```bash
pip install virtualenv
```

Once virtualenv is installed, you can create a new virtual environment with specific Python version by running:

```bash
virtualenv -p /usr/bin/python3 envname
```

You can deactivate the environment using the command `deactivate` .