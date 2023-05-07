### VECT



##### 1: 作为maven项目导入

VECT是作为一个maven项目开发的，因此您可以直接使用IntelliJ IDEA加载项目来构建环境。

##### 2: 选择测试对象

在这个项目中我们不提供测试对象，使用者可以根据自身需求下载，并且要求满足如下的文件结构。测试项目默认只扫描openjdk8、11、12、13、14五个文件夹，如果想要测试更多的JVM版本请修改`VECT\DTJVM\src\main\resources\default.properties` 中的 `openjdk.versions`。

```
├── 01JVMS
│   ├── Windows
│   |   ├── openjdk8
│   |   │   ├── hotspot_8  // including bin folder
│   |   │   ├── bisheng_8
│   |   │   └── openj9_8   
│   |   ├── openjdk11
│   |   │   ├── hotspot_11  
│   |   │   ├── bisheng_11
│   |   │   └── openj9_11 
│   |   ├── openjdk12
│   |   ├── openjdk13
│   |   └── openjdk14
│   ├── linux64
└── └── macOSx64
```

##### 3: 创建sootOuput目录

为了不破坏原始的 02benchmark 中的内容，我们使用它们的副本进行合成。请创建 sootOuput 目录，并将 02benchmark 下的所有项目复制到 sootOuput 。如果您需要添加新的 benchmark，可以直接将文件夹拷贝到 02benchmark 和 sootOuput 当中。

##### 4: 环境测试

您可以通过在 `VECT/Main/src/Main/java`下执行类 `Preview` 测试环境信息。

执行环境

```
============================================ Testing Platform Information ============================================
     os name: windows 10
  os version: 10.0
   java home: yourpath
java version: 11.0.14.1
======================================================================================================================
```

JVM测试对象信息

```
================================================= JVM Implementation =================================================
JVM root path: .\01JVMS\Windows\openjdk8
  JVM Version: openjdk8
     Java Cmd: yourpath\.\01JVMS\Windows\openjdk8\jdk_win_8_hotspot\bin\java.exe
======================================================================================================================
================================================= JVM Implementation =================================================
JVM root path: .\01JVMS\Windows\openjdk8
  JVM Version: openjdk8
     Java Cmd: yourpath\.\01JVMS\Windows\openjdk8\jdk_win_8_openj9\bin\java.exe
======================================================================================================================
```

测试 benchmark 信息

```
================================================= Project Information =================================================
Project Path: yourpath/02Benchmarks/HotspotTests-Java
Project Name: HotspotTests-Java
         lib: yourpath/02Benchmarks/HotspotTests-Java/lib
         src: yourpath/02Benchmarks/HotspotTests-Java/src
   total src: 8856
        test: null
  total test: 0
   src class: yourpath/02Benchmarks/HotspotTests-Java/out/production/HotspotTests-Java
  test class: null
 applicaiton: 3986/12134
 junit class: 26/0
=======================================================================================================================
```

##### 5. 开始测试

我们提供了项目的源文件，您可以对源代码进行更改，您可以设置-help参数来获取我们的帮助文档。





如果需要更换提供代码片段的项目，需要进行一定的调整。（因为需要重新聚类）

##### 1: 将 ingredients 转化为 java文件

需要提供三个参数

* 项目名：`HotspotTests-Java`
* 数据存储位置：`Z:\VECT_demo\clusterTest`
* 反编译工具jad位置：`Z:\VECT_demo\VECT\jad.exe`

```
java -cp VECT.jar core.CodeClusterHelper HotspotTests-Java Z:\VECT_demo\clusterTest Z:\VECT_demo\VECT\jad.exe
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

请将 `csvFile` 下生成的 `model Name+Assignment.csv` 文件复制到 `VECT ` 文件夹下

##### 4.继续测试