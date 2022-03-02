"""
  Name: load_settings.py
  Author: Sasha Sax, CVGL
  Desc: Loads runtime settings and exposes them via 'settings'

  Usage: for import only
"""

import ast
import os
import logging
import sys
import warnings
import json
from subprocess import call


def load_settings(args):
    ''' Will set the global 'settings' namespace with, in order of priority, 
        1. Any command-line arguments passed in through 'args'
        2. Any key-value pairs that are stored in a dict called 'override_settings'
            in any file called 'settings.py' which is located in the cwd or a parent.
            Files that are closer to the cwd have higher priority than files which
            are further away. 
        3. The default settings from 'import settings' 
    Args:
        args: Command line arguments. A Namespece which has key-value pairs that 
            override settings
    Returns
        settings: The mutated namespace. 
    '''
    arg_dict = parse_cmdline_args(args)

    settings_come_from = {}

    def override_settings_dict_with(settings_dict, override_dict, settings_come_from, settings_location):
        for k, v in override_dict.items():
            if k.startswith('__'): continue
            if k in settings_dict:
                settings_come_from[k] = settings_location
                if type(v) != type(settings_dict[k]):
                    try:
                        settings_dict[k] = ast.literal_eval(v)
                    except:
                        settings_dict[k] = v
                else:
                    settings_dict[k] = v

                logging.getLogger().setLevel(logging.INFO)
                logging.info(" {} = {} added to the setting...".format(k, v))
            else:
                logging.warning(" {} not defined in the setting...".format(k))
                # raise ValueError(" {} not defined in the setting...".format(k))

        return settings_dict

    # Get default settings
    import settings
    settings_dict = settings.__dict__  # Namespace()
    for k, v in settings_dict.items():
        settings_come_from[k] = '"import settings"'

    # Override settings if there is a settings.py file in the current directory
    def get_directory_hierarchy():
        ''' returns all directories up to root, root first. '''
        current_dir = os.getcwd()
        dir_order = []
        last_dir = ''
        while current_dir != last_dir:
            dir_order.append(current_dir)
            last_dir = current_dir
            current_dir = os.path.dirname(current_dir)
        dir_order.reverse()
        return dir_order

    # update the settings with local overrides
    for directory in get_directory_hierarchy():
        settingspath = os.path.join(directory, 'override_settings.py')
        if os.path.isfile(settingspath):
            sys.path.insert(0, directory)
            from override_settings import override_settings
            override_settings_dict_with(settings_dict, override_settings, settings_come_from,
                                        settings_location=directory)
            del sys.path[0]
            del override_settings
            del sys.modules["override_settings"]

    # Override settings with command-line arguments 
    override_settings_dict_with(settings_dict, arg_dict, settings_come_from, settings_location='CMDLINE')

    settings.LOGGING_LEVEL = 'INFO'
    global logger
    logger = create_logger("default_logger", filename=None, level=settings.LOGGING_LEVEL)
    settings.LOGGER = logger

    for setting, location in sorted(settings_come_from.items()):
        if setting.startswith('__') or setting.upper() != setting: continue
        logger.debug("Using {} from {}".format(setting, location))

    return settings


def create_logger(logger_name, filename=None, level=logging.INFO):
    logging.basicConfig(filename=filename)
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    return logger


def parse_cmdline_args_old(argv):
    ''' Parses all --OPTION_NAME=val into OPTION_NAME->val
    '''
    argsdict = {}
    for farg in argv:
        if farg.startswith('--') and '=' in farg:
            (arg, val) = farg.split("=")
            arg = arg[2:]
            argsdict[arg] = val
    return argsdict


def parse_cmdline_args(argv):
    ''' Parses all OPTION_NAME=val into OPTION_NAME->val
    '''
    argsdict = {}
    for farg in argv:
        if '=' in farg:
            (arg, val) = farg.split("=")
            argsdict[arg] = val
    return argsdict


def validate_blender_settings(settings_ns):
    ''' Checks all settings for internal consistency, and makes sure that the running
      version of Blender is appropriate 
    Args:
        settings: A namespace that contains the parameters from settings.py
    
    Raises:
        RuntimeError: Depends, but describes the problem
    '''
    # Check version number
    logger.debug("Python version: {}".format(sys.version))

    try:
        import bpy
        imported_bpy = True
        logger.debug("Blender version: {}".format(bpy.app.version))
    except:
        imported_bpy = False
        Warning('No Blender found.')
    if imported_bpy and bpy.app.version[1] < 78:
        raise RuntimeError('Blender version must be 2.78, but is {}.'.format(bpy.app.version))
    if settings_ns.CREATE_PANOS and settings_ns.CREATE_FIXATED:
        raise RuntimeError(
            'Cannot create both panos and non-panos in the same run. Either turn off panos or turn off fixated!')



global settings
settings = load_settings(sys.argv)
validate_blender_settings(settings)

import warnings

warnings.filterwarnings("ignore", message="is a low contrast image")
