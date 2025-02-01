import subprocess

import click
# ------------------------------------------------------------------------------

'''
Command line interface to cv-tools library
'''


@click.group()
def main():
    pass


@main.command()
def bash_completion():
    '''
    BASH completion code to be written to a _cv-tools completion file.
    '''
    cmd = '_CV_TOOLS_COMPLETE=bash_source cv-tools'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


@main.command()
def zsh_completion():
    '''
    ZSH completion code to be written to a _cv-tools completion file.
    '''
    cmd = '_CV_TOOLS_COMPLETE=zsh_source cv-tools'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


if __name__ == '__main__':
    main()
