import subprocess

import click
# ------------------------------------------------------------------------------

'''
Command line interface to cv-depot library
'''


@click.group()
def main():
    pass


@main.command()
def bash_completion():
    '''
    BASH completion code to be written to a _cv-depot completion file.
    '''
    cmd = '_cv_depot_COMPLETE=bash_source cv-depot'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


@main.command()
def zsh_completion():
    '''
    ZSH completion code to be written to a _cv-depot completion file.
    '''
    cmd = '_cv_depot_COMPLETE=zsh_source cv-depot'
    result = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE)
    result.wait()
    click.echo(result.stdout.read())


if __name__ == '__main__':
    main()
