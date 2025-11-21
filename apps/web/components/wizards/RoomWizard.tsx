'use client';

import React, { useState, useCallback } from 'react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card } from '@/components/ui/card';
import { RoomType, getRoomTypeConfig, ObjectCategory } from '@3dix/types';
import { ChevronRight, ChevronLeft, Check } from 'lucide-react';

interface RoomWizardProps {
  roomType: RoomType;
  initialData?: {
    name?: string;
    dimensions?: { width: number; length: number; height: number };
    selectedCategories?: string[];
  };
  onComplete: (data: {
    name: string;
    dimensions: { width: number; length: number; height: number };
    selectedCategories: string[];
  }) => void;
  onCancel?: () => void;
}

interface WizardStep {
  id: string;
  title: string;
  description: string;
  component: React.ReactNode;
}

export function RoomWizard({
  roomType,
  initialData,
  onComplete,
  onCancel,
}: RoomWizardProps) {
  const config = getRoomTypeConfig(roomType);
  const [currentStep, setCurrentStep] = useState(0);
  const [name, setName] = useState(initialData?.name || '');
  const [dimensions, setDimensions] = useState(
    initialData?.dimensions || config.defaultDimensions
  );
  const [selectedCategories, setSelectedCategories] = useState<string[]>(
    initialData?.selectedCategories || config.categories.filter(c => c.required).map(c => c.id)
  );

  const steps: WizardStep[] = [
    {
      id: 'name',
      title: 'Room Name',
      description: 'Give your room a name',
      component: (
        <div className="space-y-4">
          <div>
            <Label htmlFor="room-name">Room Name</Label>
            <Input
              id="room-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder={`My ${config.name}`}
              className="mt-2"
            />
          </div>
        </div>
      ),
    },
    {
      id: 'dimensions',
      title: 'Room Dimensions',
      description: 'Set the size of your room',
      component: (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <div>
              <Label htmlFor="width">Width (m)</Label>
              <Input
                id="width"
                type="number"
                step="0.1"
                min="1"
                max="20"
                value={dimensions.width}
                onChange={(e) =>
                  setDimensions({ ...dimensions, width: parseFloat(e.target.value) || 1 })
                }
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="length">Length (m)</Label>
              <Input
                id="length"
                type="number"
                step="0.1"
                min="1"
                max="20"
                value={dimensions.length}
                onChange={(e) =>
                  setDimensions({ ...dimensions, length: parseFloat(e.target.value) || 1 })
                }
                className="mt-2"
              />
            </div>
            <div>
              <Label htmlFor="height">Height (m)</Label>
              <Input
                id="height"
                type="number"
                step="0.1"
                min="2"
                max="5"
                value={dimensions.height}
                onChange={(e) =>
                  setDimensions({ ...dimensions, height: parseFloat(e.target.value) || 2.5 })
                }
                className="mt-2"
              />
            </div>
          </div>
          <p className="text-sm text-muted-foreground">
            Default: {config.defaultDimensions.width}m × {config.defaultDimensions.length}m × {config.defaultDimensions.height}m
          </p>
        </div>
      ),
    },
    {
      id: 'categories',
      title: 'Select Objects',
      description: `Choose which objects to include in your ${config.name.toLowerCase()}`,
      component: (
        <CategorySelectionStep
          categories={config.categories}
          selectedCategories={selectedCategories}
          onSelectionChange={setSelectedCategories}
        />
      ),
    },
  ];

  const canProceed = useCallback(() => {
    switch (currentStep) {
      case 0:
        return name.trim().length > 0;
      case 1:
        return dimensions.width > 0 && dimensions.length > 0 && dimensions.height > 0;
      case 2:
        // Check if all required categories are selected
        const requiredCategories = config.categories.filter(c => c.required).map(c => c.id);
        return requiredCategories.every(id => selectedCategories.includes(id));
      default:
        return true;
    }
  }, [currentStep, name, dimensions, selectedCategories, config]);

  const handleNext = useCallback(() => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(currentStep + 1);
    } else {
      onComplete({ name, dimensions, selectedCategories });
    }
  }, [currentStep, steps.length, name, dimensions, selectedCategories, onComplete]);

  const handlePrevious = useCallback(() => {
    if (currentStep > 0) {
      setCurrentStep(currentStep - 1);
    }
  }, [currentStep]);

  const currentStepData = steps[currentStep];

  return (
    <Card className="p-6 max-w-2xl mx-auto">
      {/* Progress indicator */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          {steps.map((step, index) => (
            <React.Fragment key={step.id}>
              <div className="flex items-center">
                <div
                  className={`w-8 h-8 rounded-full flex items-center justify-center ${
                    index < currentStep
                      ? 'bg-green-500 text-white'
                      : index === currentStep
                      ? 'bg-blue-500 text-white'
                      : 'bg-gray-200 text-gray-600'
                  }`}
                >
                  {index < currentStep ? (
                    <Check className="h-4 w-4" />
                  ) : (
                    <span>{index + 1}</span>
                  )}
                </div>
                {index < steps.length - 1 && (
                  <div
                    className={`h-1 w-16 mx-2 ${
                      index < currentStep ? 'bg-green-500' : 'bg-gray-200'
                    }`}
                  />
                )}
              </div>
            </React.Fragment>
          ))}
        </div>
        <div className="text-center">
          <h3 className="text-lg font-semibold">{currentStepData.title}</h3>
          <p className="text-sm text-muted-foreground">{currentStepData.description}</p>
        </div>
      </div>

      {/* Step content */}
      <div className="mb-6 min-h-[300px]">{currentStepData.component}</div>

      {/* Navigation */}
      <div className="flex items-center justify-between">
        <div>
          {onCancel && (
            <Button variant="outline" onClick={onCancel}>
              Cancel
            </Button>
          )}
        </div>
        <div className="flex gap-2">
          {currentStep > 0 && (
            <Button variant="outline" onClick={handlePrevious}>
              <ChevronLeft className="h-4 w-4 mr-1" />
              Previous
            </Button>
          )}
          <Button onClick={handleNext} disabled={!canProceed()}>
            {currentStep === steps.length - 1 ? (
              <>
                <Check className="h-4 w-4 mr-1" />
                Complete
              </>
            ) : (
              <>
                Next
                <ChevronRight className="h-4 w-4 ml-1" />
              </>
            )}
          </Button>
        </div>
      </div>
    </Card>
  );
}

interface CategorySelectionStepProps {
  categories: ObjectCategory[];
  selectedCategories: string[];
  onSelectionChange: (categories: string[]) => void;
}

function CategorySelectionStep({
  categories,
  selectedCategories,
  onSelectionChange,
}: CategorySelectionStepProps) {
  const toggleCategory = useCallback(
    (categoryId: string) => {
      if (selectedCategories.includes(categoryId)) {
        // Check if category is required
        const category = categories.find(c => c.id === categoryId);
        if (category?.required) {
          return; // Cannot deselect required category
        }
        onSelectionChange(selectedCategories.filter(id => id !== categoryId));
      } else {
        onSelectionChange([...selectedCategories, categoryId]);
      }
    },
    [selectedCategories, categories, onSelectionChange]
  );

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-3">
        {categories.map((category) => {
          const isSelected = selectedCategories.includes(category.id);
          const isRequired = category.required;

          return (
            <Card
              key={category.id}
              className={`p-4 cursor-pointer transition-all ${
                isSelected
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              } ${isRequired ? 'opacity-75' : ''}`}
              onClick={() => toggleCategory(category.id)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <h4 className="font-medium">{category.name}</h4>
                  {isRequired && (
                    <span className="text-xs text-red-600">Required</span>
                  )}
                  {category.minCount !== undefined && category.maxCount !== undefined && (
                    <p className="text-xs text-muted-foreground mt-1">
                      {category.minCount === category.maxCount
                        ? `${category.minCount} required`
                        : `${category.minCount}-${category.maxCount} allowed`}
                    </p>
                  )}
                </div>
                <div
                  className={`w-5 h-5 rounded border-2 flex items-center justify-center ${
                    isSelected
                      ? 'bg-blue-500 border-blue-500'
                      : 'border-gray-300'
                  }`}
                >
                  {isSelected && <Check className="h-3 w-3 text-white" />}
                </div>
              </div>
            </Card>
          );
        })}
      </div>
      {categories.length === 0 && (
        <p className="text-sm text-muted-foreground text-center py-8">
          No specific object categories defined for this room type.
        </p>
      )}
    </div>
  );
}
