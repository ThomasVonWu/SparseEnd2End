# ThirdParty Recipe

## Add Submodule:GoogleTest
```bash
cd path/to/SparseEnd2End
git submodule add https://github.com/google/googletest.git onboard/third_party/googletest
git commit - m "add submodule googletest"
cd  third_party/googletest
cmake -B build -S .
cmake --build build
```
<left>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="resources/images/gtest.jpg" width="500">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">If you make successfully, you will get log like this.</div>
</left>