try:
	from setuptools import setup
except ImportError:
	from distutils.core import setup

config = [
		'description': 'My Project _____________',
		'author': 'Kreger51',
		'url':'https://github.com/Kreger51/__________  '            ,
		'author_email': 'selim.belhaouane@gmail.com',
		'version': '0.1_________',
		'install_requires': ['nose'],
		'packages': ['NAME'],
		'scripts': [],
		'name': 'projectname'
]

setup(**config)