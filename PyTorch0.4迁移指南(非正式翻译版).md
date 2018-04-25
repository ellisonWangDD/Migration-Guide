写在前面：本次更新最大亮点就是支持Windows啦，这对于初学者来说是件大喜事，不用再去折腾安装学习Linux系统就能正儿八经地搞深度学习了，新特性网上到处都是，我就不赘述了，0.4版本在函数接口上与之前版本还是有些许不同的，私以为最主要的还是合并了Tensor与Variable，还有就是对数据模型迁移方式的更改。下面我根据个人理解翻译了官方给的Migration Guide，愿为pytorch推广贡献一点自己的力量，理解不对的地方烦请各位指出批评，谢谢
# PyTorch 0.40 迁移指南
欢迎阅读本指南。在这个版本中pytorch推出了很多新特性并修复了原来的BUG，给用户提供了更为便捷的函数接口。在本指南中，我们只挑重点来讲，告诉大家如何将原来的代码迁移到新的版本，以下是主要的更新特性：

* `Tensor`和`Variables`hebing
* 支持0维向量（标量）`Tensor`
* 弃坑`Volatile`方式
* 全新`Tensor`定义方式
* 指定计算设备的函数更智能了

## 合并Tensor和Variable
`torch.tensor`和`torch.autograd.Variable`现在何为一类了，更准确地讲，`torch.tensor`具备原来`Variable`的全部功能。现在`Variable`还能用，但返回的也是`torch.Tensor`类型。这意味着以后没必要使用`Variable`包裹`Tensor`数据了。
###Tensor.type()变更
使用`type()`不再返回数据类型(float,double...)了。使用`isinstance()`或`x.type()`可以查看其具体数据类型。
```python
>>>x=torch.DoubleTensor([1,1,1])
>>>print(type())#返回所属类
"<class 'torch.Tensor'>"
>>>print(x.type())
"torch.DoubleTensor"
>>>print(isinstance(x,torch.DoubleTensor))
True
```
###`autograd`现在是怎样工作的？
`requires_grad`曾是`autograd`的关键选项，现在被迁移到`Tensor`的属性，用法和之前的一样。当设置`requires_grad=True`时，`autograd`开始自动记录差分值。例如：
```python
>>>x=torch.ones(1)
>>>x.requires_grad
False
>>>y=torch.ones(1)
>>>z=x+y
>>>z.requires_grad
>>>z.backward()
RuntimeError:element 0 of tensors does not require grad and does not have a grad_fn
>>>w=torch.ones(1,requires_grap=True)
>>>w.requires_grad
True
>>>total=w+z
>>>total.requires_grad
True
>>>total.backward()
>>>w.grad
tensor([1.])
>>>z.grad==x.grad==y.grad==None
True
```
**设置`requires_grad`**<br/>
除了在初始化的时候设置外，还可以使用`my_tensor.requires_grad()`来设置
###`.data`用法
在之前的版本，使用`.data`将Variable转化为Tensor。现在合并之后，调用`y=x.data`
后，`y`是另一个`Tensor`，只是与`x`共用数据部分，但在计算的求导信息与原来的`x`独立，
但是，在某些情况使用`.data`欠妥。任何`x.data`的数据变化都不会影响到`x`的梯度。更为保险的方法是使用`x.detach()`，它返回的也是与原变量共享数据的`Tensor`
##支持0维Tensor(标量)
之前版本中，求一维`Tensor`的索引返回一个数值，但是一维`Variable`却返回`(1,)`!相似的情况同样出现在求和函数中，例如`tensor.sum()`返回一个数值，然而`Variable.sum()`返回的是(1,)
还好，本次更新后pytorch支持标量了！标量可以直接用`torch.tensor`创建就像`numpy.array`那样
```python
>>>torch.tensor(3.1416)
tensor(3.1416)
>>>torch.tensor(3.1416).size()
torch.Size([])#表明这是0维数据，即标量
>>>torch.tensor([3]).size()
torch.Size([1])
>>>vector=torch.arange(2,6)
>>>vector
tensor([2.,3.,4.,5.])
>>>vector.size()
torch.Size([4])
>>>vector[3].item()
5.0
>>>mysum=torch.tensor([2,3]).sum()
>>>mysum
tensor(5)
>>>mysum.size()
torch.Size([])
```
>个人理解：新版本支持标量了，可以直接用，不像原来单个数据还给搞出个一维数组

**损失积累**<br/>
之前都使用`total_loss+=loss.data[0]`累积损失率。在0.4版本中有0维的标量了，直接用`loss.item()`得到其loss的数值就可以了。
##反对使用`volatile`选项
`volatile`选项在0.4版本中不推荐使用了，之前版本中给变量设置`volatile=True`一遍其不求导计算。现在这个功能被其他函数替代
`torch.no_grad(),torch.set_grad_enabled()`
```python
>>>x=torch.zeros(1,requires_grad=True)
>>>with torch.no_grad():
        y=x*2
>>>y.requires_grad
False
>>>is_train=False
>>>with torch.set_grad_enabled(is_train):
        y=x*2
>>>y.requires_grad
True
>>>torch.set_grad_enabled(False)
>>>y=x*2
>>>y.requires_grad
False
```
##`dtypes`,`devices`变更
在0.40版本中，使用`torch.dtype`,`torch.device`和`torch.layout`类来分配管理数据设备类型
###`torch.dtype`
以下是可用的数据类型表和它相应的tensor类型
###`torch.device`
`torch.device`包含两种设备类型，cpu和cuda。对于GPU还可以选择设备编号，例如torch.device('{设备类型}：{设备编号}')，如果不确定设备编号，默认使用`torch.device('cuda')`就会默认调用当前的显卡。可以使用`torch.cuda.current_device()`查看当前显卡
###`torch.layout`
`torch.layout`代表tensor数据配置
### 创建Tensor
在新版本中创建Tensor需要考虑dtype,device,layout和requires_grad，例如
```
>>>device=torch.device('cuda:1')
>>>x-torch.randn(3,3,dtype=torch.float64,device=device)
tensor([-0.6344,0.8534,-1.2354],
[0.8414,1.7962,1.0589],
[-0.1369,-1.0462,-0.4373],dtype=torch.float64,device='cuda:1')
>>>x.requires_grad
False
>>>x=torch.zeros(3,requires_grad=True)
>>>x.requires_grad
True
```
torch.tensor(data,...)

##模型数据迁移
在之前的版本中，当不确定计算设备(cpu or which GPU?)情况时不太好写代码。
0.4版本做出了如下更新

*  使用`to`方法可以轻松转换训练的网络和数据到不同设备之间
*  `device`属性用来指定使用的计算设备
示例demo：
```python
device=torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
input=data.to(device)#直接指定数据到哪个设备中
model=MyModule().to(device)#同样，网络模型转换到指定设备中
```
















