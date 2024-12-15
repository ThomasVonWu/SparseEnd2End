# ThirdParty Recipe

## Add Submodule:googletest
```bash
cd path/to/SparseEnd2End
git submodule add https://github.com/google/googletest.git onboard/third_party/googletest
git commit - m "add submodule googletest"
cd  onboard/third_party/googletest
cmake -B build -S .
cmake --build build
```
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/images/gtest.jpg" width="620">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">If you make successfully, you will get log like this.</div>
</center>

## Add Submodule:glog
```bash
cd path/to/SparseEnd2End
git submodule add https://github.com/google/glog.git onboard/third_party/glog
git commit - m "add submodule glog"
cd  onboard/third_party/glog
cmake -B build -S .
cmake --build build
```
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/images/glog.jpg" width="800">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">If you make successfully, you will get log like this.</div>
</center>

## Add Submodule:yaml-cpp
```bash
cd path/to/SparseEnd2End
git submodule add https://github.com/jbeder/yaml-cpp.git onboard/third_party/yaml-cpp
git commit - m "add submodule yaml-cpp"
cd  onboard/third_party/yaml-cpp
cmake -B build -S .
cmake --build build
```
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/images/yaml-cpp.jpg" width="800">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 1px;">If you make successfully, you will get log like this.</div>
</center>