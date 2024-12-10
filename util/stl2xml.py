import numpy as np
import xml.etree.ElementTree as ET
from stl import mesh
import xml.dom.minidom as minidom

def stl_to_xml(stl_path, xml_path, include_metadata=True, pretty_print=True):
    try:
        # 读取STL文件
        mesh_data = mesh.Mesh.from_file(stl_path)
        
        # 创建根元素
        root = ET.Element("model")
        
        # 添加元数据
        if include_metadata:
            metadata = ET.SubElement(root, "metadata")
            ET.SubElement(metadata, "vertices_count").text = str(len(mesh_data.vectors.flatten()) // 3)
            ET.SubElement(metadata, "faces_count").text = str(len(mesh_data.vectors))
            
        # 创建几何数据部分
        geometry = ET.SubElement(root, "geometry")
        
        # 处理面片数据
        faces = ET.SubElement(geometry, "faces")
        
        # 存储顶点和法向量
        for i, (triangle, normal) in enumerate(zip(mesh_data.vectors, mesh_data.normals)):
            face = ET.SubElement(faces, "face", id=str(i))
            
            # 顶点
            vertices = ET.SubElement(face, "vertices")
            for j, vertex in enumerate(triangle):
                v = ET.SubElement(vertices, "v", id=str(j))
                v.set("x", f"{vertex[0]:.6f}")
                v.set("y", f"{vertex[1]:.6f}")
                v.set("z", f"{vertex[2]:.6f}")
            
            # 法向量
            n = ET.SubElement(face, "normal")
            n.set("x", f"{normal[0]:.6f}")
            n.set("y", f"{normal[1]:.6f}")
            n.set("z", f"{normal[2]:.6f}")
        
        # 创建XML树
        tree = ET.ElementTree(root)
        
        if pretty_print:
            # 格式化XML输出
            xmlstr = minidom.parseString(ET.tostring(root)).toprettyxml(indent="    ")
            with open(xml_path, "w", encoding="utf-8") as f:
                f.write(xmlstr)
        else:
            # 直接写入
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)
            
        return True
    
    except Exception as e:
        print(f"转换错误: {str(e)}")
        return False

def validate_stl(stl_path):
    """验证STL文件"""
    try:
        mesh_data = mesh.Mesh.from_file(stl_path)
        return True
    except Exception:
        return False

# 使用示例
if __name__ == "__main__":
    input_stl = "data/xml/gripper.stl"
    output_xml = "data/output/output.xml"
    
    if validate_stl(input_stl):
        if stl_to_xml(input_stl, output_xml):
            print("转换成功！")
    else:
        print("无效的STL文件！")