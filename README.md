# 稀疏矩阵与向量乘法的优化目录
## 目录文件说明
* src/spamatrixmul.cu 矩阵乘向量各优化算法主程序文件
* src/Makefile 交叉编译文件
* introdution.pdf 优化思路说明
* intro.md 思路说明源文件
* pic/ 引用截图或图片
* test.xlsx 测试结果统计文件
* README.md 目录文件
## 使用方式
* 1.编译：直接在根目录下运行make指令即可完成编译。
* 2.运行main文件，即可得到运行结果。
## 运行结果说明
* 1、运行时间结果（分为五个优化级别的结果，原来没有优化的cpu结果未再打出）
* 2、计算正确性结果（1~4分别代表三种优化和cublas库结果，均与cpu计算结果比对）
  