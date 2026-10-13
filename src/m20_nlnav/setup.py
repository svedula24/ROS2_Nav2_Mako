from setuptools import setup
package_name = 'm20_nlnav'
setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    description='Natural-language navigation: text -> resolved goals -> RunMission (m20_missions).',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['nlnav_node = m20_nlnav.nlnav_node:main']},
)
