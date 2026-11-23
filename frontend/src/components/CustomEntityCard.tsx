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
import { TagList, ENTITY_ACCENTS } from './cards';
import ProxyConnectButton from './ProxyConnectButton';

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

const ACCENT = ENTITY_ACCENTS.custom;

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
      return 'bg-success-100 text-success-800 dark:bg-success-900/30 dark:text-success-400 border border-success-200 dark:border-success-700';
    case 'group-restricted':
      return 'bg-info-100 text-info-800 dark:bg-info-900/30 dark:text-info-400 border border-info-200 dark:border-info-700';
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300 border border-gray-200 dark:border-gray-600';
  }
}

/**
 * Compact card for a user-defined custom entity record. Custom types have no
 * fixed brand color, so they use the neutral accent. The body is descriptor-
 * driven (only fields flagged show_in_list are rendered), which is this card's
 * entity-specific behavior; tag rendering is shared via TagList.
 */
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
    <div className="card hover:border-line-strong p-4 flex flex-col gap-3 hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-2">
        <h3 className="text-base font-semibold text-fg truncate">
          {record.name}
        </h3>
        <div className="flex items-center gap-1 flex-shrink-0">
          {record.is_proxied && (
            <span
              className="px-2 py-0.5 text-xs font-semibold bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-300 rounded-full border border-info-200 dark:border-info-600"
              title={
                record.proxy_client_url
                  ? `Clients connect at ${record.proxy_client_url}` +
                    (record.proxy_target_url ? ` (forwards to ${record.proxy_target_url})` : '')
                  : 'Served through the gateway proxy'
              }
            >
              Proxied
            </span>
          )}
          <span
            className={`px-2 py-0.5 text-xs font-semibold rounded-full flex items-center gap-1 ${visibilityColor(
              record.visibility,
            )}`}
          >
            {visibilityIcon(record.visibility)}
            {record.visibility}
          </span>
        </div>
      </div>

      {record.description && (
        <p className="text-sm text-fg-muted line-clamp-2">
          {record.description}
        </p>
      )}

      {listFields.length > 0 && (
        <dl className="grid grid-cols-1 gap-1 text-sm">
          {listFields.map((field) => (
            <div key={field.name} className="flex gap-2">
              <dt className="text-fg-subtle flex-shrink-0">
                {labelFor(field)}:
              </dt>
              <dd className="text-fg truncate">
                {formatValue(field, record.attributes[field.name])}
              </dd>
            </div>
          ))}
        </dl>
      )}

      <TagList tags={record.tags} accent={ACCENT} prefix="#" className="gap-1" />

      <div className="flex items-center justify-between gap-1 mt-auto pt-2 border-t border-line">
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
          {/* Always rendered: a disabled Connect explains that the record is not
              routed through the gateway, which an absent control cannot. */}
          <ProxyConnectButton
            clientUrl={record.proxy_client_url}
            targetUrl={record.proxy_target_url}
            connectNotes={record.proxy_connect_notes}
          />
          <button
            type="button"
            onClick={() => onView(record)}
            className="btn-icon"
            aria-label="View details"
          >
            <InformationCircleIcon className="h-4 w-4" />
          </button>
          {canModify && (
            <>
              <button
                type="button"
                onClick={() => onEdit(record)}
                className="btn-icon"
                aria-label="Edit"
              >
                <PencilIcon className="h-4 w-4" />
              </button>
              <button
                type="button"
                onClick={() => onDelete(record)}
                className="btn-icon hover:text-danger-500"
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
