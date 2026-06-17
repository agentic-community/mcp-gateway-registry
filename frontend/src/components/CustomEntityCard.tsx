import React from 'react';
import {
  PencilIcon,
  TrashIcon,
  GlobeAltIcon,
  LockClosedIcon,
  UserGroupIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import {
  CustomEntityRecord,
  CustomFieldDescriptor,
  CustomTypeDescriptor,
} from '../types/customEntity';
import { labelFor } from '../utils/humanize';
import StarRatingWidget from './StarRatingWidget';

interface CustomEntityCardProps {
  descriptor: CustomTypeDescriptor;
  record: CustomEntityRecord;
  canModify: boolean;
  onView: (record: CustomEntityRecord) => void;
  onEdit: (record: CustomEntityRecord) => void;
  onDelete: (record: CustomEntityRecord) => void;
  authToken?: string | null;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}

/** Render an attribute value as a TEXT NODE only (no HTML injection). */
export function formatValue(
  field: CustomFieldDescriptor,
  value: unknown,
): string {
  if (value === undefined || value === null || value === '') return '—';
  switch (field.datatype) {
    case 'bool':
      return value ? 'Yes' : 'No';
    case 'array<string>':
      return Array.isArray(value) ? (value as string[]).join(', ') : String(value);
    default:
      return String(value);
  }
}

function visibilityIcon(visibility: string) {
  switch (visibility) {
    case 'public':
      return <GlobeAltIcon className="h-3 w-3" />;
    case 'group-restricted':
      return <UserGroupIcon className="h-3 w-3" />;
    default:
      return <LockClosedIcon className="h-3 w-3" />;
  }
}

function visibilityColor(visibility: string): string {
  switch (visibility) {
    case 'public':
      return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400 border border-green-200 dark:border-green-700';
    case 'group-restricted':
      return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400 border border-blue-200 dark:border-blue-700';
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600';
  }
}

const CustomEntityCard: React.FC<CustomEntityCardProps> = ({
  descriptor,
  record,
  canModify,
  onView,
  onEdit,
  onDelete,
  authToken,
  onShowToast,
}) => {
  const listFields = descriptor.fields.filter((f) => f.show_in_list);

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 p-4 flex flex-col gap-3 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-base font-semibold text-gray-900 dark:text-white truncate">
          {record.name}
        </h3>
        <span
          className={`px-2 py-0.5 text-xs font-semibold rounded-full flex-shrink-0 flex items-center gap-1 ${visibilityColor(
            record.visibility,
          )}`}
        >
          {visibilityIcon(record.visibility)}
          {record.visibility}
        </span>
      </div>

      {record.description && (
        <p className="text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
          {record.description}
        </p>
      )}

      {listFields.length > 0 && (
        <dl className="grid grid-cols-1 gap-1 text-sm">
          {listFields.map((field) => (
            <div key={field.name} className="flex gap-2">
              <dt className="text-gray-500 dark:text-gray-400 flex-shrink-0">
                {labelFor(field)}:
              </dt>
              <dd className="text-gray-900 dark:text-white truncate">
                {formatValue(field, record.attributes[field.name])}
              </dd>
            </div>
          ))}
        </dl>
      )}

      {record.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {record.tags.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="px-2 py-0.5 text-xs rounded-full bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300"
            >
              {tag}
            </span>
          ))}
          {record.tags.length > 3 && (
            <span className="px-2 py-0.5 text-xs text-gray-400">
              +{record.tags.length - 3}
            </span>
          )}
        </div>
      )}

      <div className="flex items-center justify-between gap-1 mt-auto pt-2 border-t border-gray-100 dark:border-gray-700">
        <StarRatingWidget
          resourceType="custom"
          path={record.path}
          initialRating={record.num_stars}
          initialCount={record.rating_details?.length ?? 0}
          ratingDetails={record.rating_details}
          authToken={authToken}
          onShowToast={onShowToast}
        />
        <div className="flex items-center gap-1">
        <button
          type="button"
          onClick={() => onView(record)}
          className="p-2 text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 rounded-lg transition-colors"
          aria-label="View details"
        >
          <InformationCircleIcon className="h-4 w-4" />
        </button>
        {canModify && (
          <>
            <button
              type="button"
              onClick={() => onEdit(record)}
              className="p-2 text-gray-400 hover:text-teal-600 dark:hover:text-teal-400 rounded-lg transition-colors"
              aria-label="Edit"
            >
              <PencilIcon className="h-4 w-4" />
            </button>
            <button
              type="button"
              onClick={() => onDelete(record)}
              className="p-2 text-gray-400 hover:text-red-500 rounded-lg transition-colors"
              aria-label="Delete"
            >
              <TrashIcon className="h-4 w-4" />
            </button>
          </>
        )}
        </div>
      </div>
    </div>
  );
};

export default CustomEntityCard;
