### 使用 VECT 进行测试

##### 1: 将VECT文件夹作为maven项目导入

您可以进入 `VECT` 文件夹直接使用 `VECT.jar` 进行测试，也可以直接使用 `IntelliJ IDEA` 加载项目来构建环境，从而修改项目源代码。

##### 2: 下载测试对象

在这个项目中我们不提供测试对象，使用者可以根据自身需求下载，并且要求满足如下的文件结构。测试项目默认只扫描openjdk8、11、17、20四个文件夹，如果想要测试更多的JVM版本请修改`VECT\DTJVM\src\main\resources\default.properties` 中的 `openjdk.versions`。

```
├── 01JVMS
│   ├── linux64
│   |   ├── openjdk8
│   |   │   ├── hotspot_8  // including bin folder
│   |   │   ├── bisheng_8
│   |   │   └── openj9_8   
│   |   ├── openjdk11
│   |   │   ├── hotspot_11  
│   |   │   ├── bisheng_11
│   |   │   └── openj9_11 
│   |   ├── openjdk17
│   |   │   ├── hotspot_17 
│   |   │   ├── bisheng_17
│   |   │   └── openj9_17
└── └── └── openjdk20
```

##### 3: 创建sootOuput目录

为了不破坏原始的 02benchmark 中的内容，我们使用它们的副本进行合成。请创建 sootOuput 目录，并将 02benchmark 下的所有项目复制到 sootOuput 。如果您需要添加新的 benchmark，可以直接将文件夹拷贝到 02benchmark 和 sootOuput 当中。

##### 4: 执行环境文件结构

```
VECT
├── 01JVMS                      :  测试对象
├── 02Benchmarks                :  原始 benchmarks
├── 03results                   :  存放差异测试报告
├── 04SynthesisHistory          :  合成过程中生成的测试程序 
├── sootOutput                  :  用于合成的 benchmarks
├── lib                         :  存放需要用到的 jar 包
├── VECT.jar                    :  项目 jar 包
└── PlBartAssignment.csv        :  聚类所需要的文件
```

##### 5.环境测试

您可以通过在 `VECT/Main/src/Main/java`下执行类 `Preview` 测试环境信息。

```
java -cp VECT.jar Preview
```

执行环境

```
======================================== Testing Platform Information ===================================
     os name: linux
  os version: 5.15.0-69-generic
   java home: /usr/lib/jvm/java-8-openjdk-amd64/jre
java version: 1.8.0_292
=========================================================================================================
```

JVM测试对象信息

```
======================================== JVM Implementation ============================================
JVM root path: ./01JVMS/linux64/openjdk8
  JVM Version: openjdk8
    Java Cmd: yourpath/./01JVMS/linux64/openjdk8/jdk_linux_8_bisheng/bin/java
========================================================================================================
======================================== JVM Implementation ============================================
JVM root path: ./01JVMS/linux64/openjdk11
  JVM Version: openjdk11
    Java Cmd: yourpath/./01JVMS/linux64/openjdk11/jdk_linux_11_bisheng/bin/java
========================================================================================================
```

测试 benchmark 信息

```
======================================== Project Information ===========================================
Project Path: yourpath/./02Benchmarks/HotspotTests-Java
Project Name: HotspotTests-Java
         lib: yourpath/./02Benchmarks/HotspotTests-Java/lib
         src: yourpath/./02Benchmarks/HotspotTests-Java/src
   total src: 8856
        test: null
  total test: 0
   src class: yourpath/./02Benchmarks/HotspotTests-Java/out/production/HotspotTests-Java
  test class: null
 applicaiton: 3986/12135
 junit class: 26/0
========================================================================================================
======================================== Project Information ===========================================
Project Path: yourpath/./02Benchmarks/templateClass
Project Name: templateClass
         lib: null
         src: null
   total src: 0
        test: null
  total test: 0
   src class: yourpath/./02Benchmarks/templateClass/out/production/templateClass
  test class: null
 applicaiton: 1/1
 junit class: 0/0
========================================================================================================
```

##### 6.开始测试

我们提供了项目的源文件以及 jar 包，您可以对修改源代码以自定义内容。

工具的入口为 Main，您可以使用如下指令设置-help参数来获取我们的帮助文档：

```
java -cp VECT.jar:lib/rt.jar Main -help
```

下面是一个进行测试的例子：

```
java -cp VECT.jar:lib/rt.jar Main -s HotspotTests-Java -p HotspotTests-Java -sl rws -cl plbart -ch true -et 86000
```





### 为JVM设置执行时参数

您可以在目标JVM的文件夹下创建文件`.options` ，程序在执行时会自动读取文件中的参数设置。

下面以为 Bisheng JDK17设置参数为例子进行说明：

* 在 `./01JVMS/linux64/openjdk17/bisheng_17` 下创建文件 `.options`

* 向文件中写入参数设置

  ```txt
  --illegal-access=warn
  --add-opens=java.base/java.lang=ALL-UNNAMED
  --add-opens=java.base/java.util.concurrent=ALL-UNNAMED
  --add-opens=java.base/jdk.internal.platform=ALL-UNNAMED
  --add-opens=jdk.jartool/sun.tools.jar=ALL-UNNAMED
  --add-opens=java.base/javax.crypto.spec=ALL-UNNAMED
  --add-opens=java.base/jdk.internal.org.objectweb.asm=ALL-UNNAMED
  --add-opens=java.desktop/java.awt=ALL-UNNAMED
  ```

* 正常进行测试

### 初步去重功能的使用

项目提供了测试结果进行初步去重的功能，您可以通过执行

```
java -cp VECT.jar:lib/rt.jar core.DiffLogAnalyzer yourPathTo/difference.log
```

 进行去重。去重后的结果会被放在对应文件夹中，其中`uniqueCrash.txt`代表异常相关的测试用例，`checksum.txt`代表与输出相关的测试用例。



### 更改 VECT 提供种子文件的项目

假设新的种子项目名为 `NewProjectName` 。您需要按照如下的结构编译项目：

```
NewProjectName
├── src                       :  源码
├── out                       :  编译后的类文件
│   ├── production
│   |   ├── NewProjectName
│   |   │   ├── 相关类文件
├── lib                       :  执行所需的 jar 包
├── testcases.txt             :  被使用的种子 (即可以不使用全部的种子)
└── skipclass.txt             :  不进行 checksum 的种子 (因为某些种子因随机等因素产生误报)
```

将项目文件夹分别保存到 `02Benchmarks` `sootOutput` 当中，并且在执行时设置 `-s NewProjectName` 的参数使用新项目。







### 更改 VECT 提供代码片段的项目

如果需要更换提供代码片段的项目，需要进行一定的调整。（因为需要重新聚类）

##### 1: 将 ingredients 转化为 java文件

需要提供三个参数

* 项目名：`HotspotTests-Java`
* 数据存储位置：`../clusterTest`
* 反编译工具jad位置：`./jad`

```
java -cp VECT.jar core.CodeClusterHelper HotspotTests-Java ../clusterTest ./jad
```

##### 2.代码片段向量化

这里提供了四种代码向量化方式，可以根据需求使用

```
python InferCodeVector.py
python CodeT5Vector.py
python CodeBERTVector.py
python PlBartVector.py
```

##### 3.代码片段聚类

聚类需要先执行层次聚类，然后再对数据进行补全。根据代码向量化的方法，有四个参数可以选择，分别是 InferCode、CodeT5、CodeBERT 和 PlBart

```
python hierarchicalClustering.py PlBart
python assignmentComplete.py PlBart
```

请将 `csvFile` 下生成的 `model Name+Assignment.csv` 文件复制到 `VECT ` 文件夹下，后续正常进行测试即可
