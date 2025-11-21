import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Input } from '@/components/ui/input';

export interface MaskControlsState {
  maskType: 'none' | 'floor' | 'arch';
  maskUrl: string;
}

interface MaskControlsProps {
  value: MaskControlsState;
  onChange: (value: MaskControlsState) => void;
}

export function MaskControls({ value, onChange }: MaskControlsProps) {
  return (
    <div className="space-y-3">
      <div className="space-y-1">
        <Label>Mask Type</Label>
        <Select
          value={value.maskType}
          onValueChange={(val: any) => onChange({ ...value, maskType: val })}
        >
          <SelectTrigger>
            <SelectValue placeholder="Mask type" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="none">None (unconditional)</SelectItem>
            <SelectItem value="floor">Floor mask</SelectItem>
            <SelectItem value="arch">Architecture (floor/door/window)</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="space-y-1">
        <Label>Mask Image URL (optional)</Label>
        <Input
          placeholder="https://..."
          value={value.maskUrl}
          onChange={(e) => onChange({ ...value, maskUrl: e.target.value })}
        />
      </div>
    </div>
  );
}
