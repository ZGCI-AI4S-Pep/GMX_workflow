import re

def format_pdb_file(pdb_file):
    formatted_lines = []
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                # 确保原子名与残基名之间有空格
                line = re.sub(r'(\S)(MOL\b)', r'\1 \2', line)
                line = re.sub(r'(\bMOL)(\S)', r'\1 \2', line)
                formatted_lines.append(line)
            else:
                formatted_lines.append(line)
    with open(pdb_file, 'w') as file:
        file.writelines(formatted_lines)

def read_pdb_atoms(pdb_file):
    pdb_atoms = {}
    with open(pdb_file, 'r') as file:
        for line in file:
            if line.startswith('ATOM'):
                parts = line.split()
                atom_id = int(parts[1])  # 第2列为原子ID
                atom_name = parts[2]     # 第3列为原子名
                pdb_atoms[atom_id] = atom_name
    return pdb_atoms

def read_gro_atoms(gro_file):
    gro_atoms = []
    gro_numbers = []
    gro_lines = []
    with open(gro_file, 'r') as file:
        lines = file.readlines()
        for i, line in enumerate(lines):
            if i < 2:  # 跳过前两行（标题和原子数）
                gro_lines.append(line)
                continue
            if not line.strip():  # 跳过空行
                gro_lines.append(line)
                continue
            if i == len(lines) - 1:  # 保留最后一行（盒子信息）
                gro_lines.append(line)
                continue
            # 分割字段，最少需要6个字段（残基号+残基名、原子名、原子号、x, y, z）
            parts = line.split()
            if len(parts) < 6:
                print(f"Warning: Line {i+1} has invalid format: {line.strip()}")
                gro_lines.append(line)
                continue
            residue_info = parts[0]  # 残基号+残基名
            atom_name = parts[1]     # 原子名
            atom_number = parts[2]   # 原子号
            x = parts[3]             # x坐标
            y = parts[4]             # y坐标
            z = parts[5]             # z坐标
            gro_atoms.append(atom_name)
            gro_numbers.append(atom_number)
            gro_lines.append(line)
    return gro_atoms, gro_numbers, gro_lines

def write_gro_file(gro_file, lines):
    with open(gro_file, 'w') as file:
        file.writelines(lines)

def correct_gro_atoms(pdb_atoms, gro_atoms, gro_numbers, gro_lines):
    corrected_lines = gro_lines[:2]  # 保留前两行
    for i, (atom_name, atom_number) in enumerate(zip(gro_atoms, gro_numbers)):
        original_line = gro_lines[i + 2].rstrip('\n')
        parts = original_line.split()
        if len(parts) < 6:
            corrected_lines.append(gro_lines[i + 2])
            continue
        try:
            atom_id = int(atom_number)
        except ValueError:
            print(f"Warning: Invalid atom number '{atom_number}' in line {i+3}.")
            corrected_lines.append(gro_lines[i + 2])
            continue
        if atom_id in pdb_atoms:
            new_atom_name = pdb_atoms[atom_id]
            # 替换原子名（确保不超过5字符）
            new_atom_name = new_atom_name.ljust(5)[:5]
            # 重新拼接行，保留后续坐标部分
            corrected_line = f"{parts[0]:<10}{new_atom_name:>5}{parts[2]:>5} {float(parts[3]):>8.3f} {float(parts[4]):>8.3f} {float(parts[5]):>8.3f}"
            corrected_lines.append(corrected_line + '\n')
        else:
            print(f"Warning: Atom ID {atom_id} not found in PDB.")
            corrected_lines.append(gro_lines[i + 2])
    # 添加最后一行（盒子信息）
    corrected_lines.append(gro_lines[-1])
    return corrected_lines

def main():
    pdb_file = 'exam.pdb'
    gro_file = 'exam.gro'
    
    format_pdb_file(pdb_file)  # 格式化PDB中MOL
    
    pdb_atoms = read_pdb_atoms(pdb_file)
    gro_atoms, gro_numbers, gro_lines = read_gro_atoms(gro_file)
    
    corrected_lines = correct_gro_atoms(pdb_atoms, gro_atoms, gro_numbers, gro_lines)
    
    write_gro_file(gro_file, corrected_lines)
    print("GRO file corrected with PDB atom names.")

if __name__ == "__main__":
    main()