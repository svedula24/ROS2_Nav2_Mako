from setuptools import setup
package_name = 'm20_vlmnav'
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
    description='Real-time B (NaVILA-style): a VLM picks the next short move; Nav2 behaviors execute it.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={'console_scripts': ['vlmnav_node = m20_vlmnav.vlmnav_node:main']},
)
