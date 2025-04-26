def modify_residue_names(mol2_file, output_file):
    with open(mol2_file, 'r') as file:
        lines = file.readlines()

    modified_lines = []
    in_atom_section = False

    for line in lines:
        if line.startswith('@<TRIPOS>ATOM'):
            in_atom_section = True
            modified_lines.append(line)
            continue
        if in_atom_section and line.strip().isdigit():
            in_atom_section = False
            modified_lines.append(line)
            continue
        if in_atom_section:
            parts = line.split()
            if len(parts) >= 8:
                # 修改残基名称为 MOL
                parts[7] = 'MOL'
                # 重新格式化行，确保每一列之间有适当的空格
                modified_line = f"{int(parts[0]):>5} {parts[1]:<2} {float(parts[2]):>10.4f} {float(parts[3]):>10.4f} {float(parts[4]):>10.4f} {parts[5]:<4} {int(parts[6]):>2} {parts[7]:<3} {float(parts[8]):>10.4f}\n"
                modified_lines.append(modified_line)
            else:
                modified_lines.append(line)
        else:
            modified_lines.append(line)

    with open(output_file, 'w') as file:
        file.writelines(modified_lines)

def main():
    mol2_file = 'exam.mol2'
    output_file = 'exam.mol2'
    
    modify_residue_names(mol2_file, output_file)
    print(f"Residue names in {mol2_file} have been modified and saved to {output_file}.")

if __name__ == "__main__":
    main()